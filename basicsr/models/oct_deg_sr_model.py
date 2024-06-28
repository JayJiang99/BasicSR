import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale

from basicsr.models import lr_scheduler as lr_scheduler

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from .base_model import BaseModel

logger = logging.getLogger("base")

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, 0, 1)
        output = (output * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


@MODEL_REGISTRY.register()
class OCTDegSRModel(BaseModel):
    def __init__(self, opt):
        super(OCTDegSRModel, self).__init__(opt)
        if self.opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # TODO: should be adjusted
        self.data_names = ["lq", "gt"]
        

        # define network net_g
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        
        if self.is_train:
            self.init_training_settings()
            train_opt = self.opt['train']
            self.max_grad_norm = train_opt["max_grad_norm"]
            self.quant = Quantization()
            self.D_ratio = train_opt["D_ratio"]
            self.optim_sr = train_opt["optim_sr"]
            self.optim_deg = train_opt["optim_deg"]
            # self.gray_dis = train_opt["gray_dis"]

            ## buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])

            
    def init_training_settings(self):
        train_opt = self.opt['train']

         # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define networks for training
        self.network_names = ["network_deg", "network_dis1", "network_dis2"]
        self.networks = {}
        nets_opt = self.opt['networks']
        
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))
        for name in defined_network_names:
            setattr(self, name, build_network(nets_opt[name]))
            # self.networks[name] = getattr(self, name)
            self.networks[name] = self.model_to_device(getattr(self, name))
            self.print_network(self.networks[name])
            
        # TODO: add how to load_networks


        self.set_network_state(self.networks.keys(), "train")
        self.net_g.train()
        self.net_g_ema.eval()

        # define losses
        self.losses = {}
        
        self.loss_names = [
            "lr_adv",
            "sr_adv",
            "sr_pix_trans",
            "sr_pixel",
            "sr_perceptual",
            "lr_quant",
            "lr_gauss",
            "noise_mean",
            "color"
        ]
        loss_opt = train_opt["losses"]
        defined_loss_names = list(loss_opt.keys())
        assert set(defined_loss_names).issubset(set(self.loss_names))
        for name in defined_loss_names:
            self.losses[name] = build_loss(loss_opt[name]).to(self.device)
        self.optimizers = {}
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        # build optmizers
        train_opt = self.opt['train']
        optimizer_opts = train_opt["optimizers"]

        if "default" in optimizer_opts.keys():
            default_optim = optimizer_opts.pop("default")

        defined_optimizer_names = list(optimizer_opts.keys())
        # TODO:
        # print(optimizer_opts.keys())
        
        print(defined_optimizer_names)
        # TODO: fix it later
        # assert set(defined_networks_names).issubset(optimizer_opts.keys())

        for name in defined_optimizer_names:
            optim_opt = optimizer_opts[name]
            if optim_opt is None:
                optim_opt = default_optim.copy()

            params = []
            print(name)
            if name == 'net_g':
                for v in self.net_g.parameters():
                    if v.requires_grad:
                        params.append(v)
            else:
                for v in self.networks[name].parameters():
                    if v.requires_grad:
                        params.append(v)
            optim_type = optim_opt.pop("type")
            optimizer = getattr(torch.optim, optim_type)(params=params, **optim_opt)
            self.optimizers[name] = optimizer
    def setup_schedulers(self):
        
        
        """Set up schedulers."""
        train_opt = self.opt['train']
        optimizer_opts = train_opt["optimizers"]
        defined_optimizer_names = list(optimizer_opts.keys())
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for name in defined_optimizer_names:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(self.optimizers[name], **train_opt['scheduler']))
            # for optimizer in self.optimizers:
            #     self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for name in defined_optimizer_names:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(self.optimizers[name], **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
    
    def feed_data(self, data):

        self.syn_hr = data["gt"].to(self.device)
        self.real_lr = data["lq"].to(self.device)

    def deg_forward(self):
        # degrade the input hr to fake lr with network_deg
        (
            self.fake_real_lr,
            self.predicted_kernel,
            self.predicted_noise,
         ) = self.network_deg(self.syn_hr)
        # if self.fake_real_lr.size(1) == 1:
        #     # repeat it to be 3 channel for the sr model
        #     self.fake_real_lr = self.fake_real_lr.repeat(1,3,1,1)
        # use net_g to get the SR image from the fake lr
        if self.losses.get("sr_pix_trans"):
            self.fake_real_lr_quant = self.quant(self.fake_real_lr)
            self.syn_sr = self.net_g(self.fake_real_lr_quant)
            # if self.syn_sr.size(1) == 3:
            #     self.syn_sr = rgb_to_grayscale(self.syn_sr)
    
    def sr_forward(self):
        # Similar to the above deg_forward. WHY TODO:
        if not self.optim_deg:
            (
                self.fake_real_lr,
                self.predicted_kernel,
                self.predicted_noise,
            ) = self.network_deg(self.syn_hr)
        # if self.fake_real_lr.size(1) == 1:
        #     # repeat it to be 3 channel for the sr model
        #     self.fake_real_lr = self.fake_real_lr.repeat(1,3,1,1)

        self.fake_real_lr_quant = self.quant(self.fake_real_lr)
        self.syn_sr = self.net_g(self.fake_real_lr_quant.detach())
        # if self.syn_sr.size(1) == 3:
        #         self.syn_sr = rgb_to_grayscale(self.syn_sr)
    
    def optimize_trans_models(self, step, loss_dict):

        self.set_requires_grad(["network_deg"], True)
        self.deg_forward()
        loss_G = 0

        if self.losses.get("lr_adv"):
            self.set_requires_grad(["network_dis1"], False)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            # Calculate the adv loss of being real between the fake lr and real lr
            g1_adv_loss = self.calculate_gan_loss_G(
                self.network_dis1, self.losses["lr_adv"], real, fake
            )
            loss_dict["g1_adv"] = g1_adv_loss.item()
            # multiply the loss weight
            loss_G +=  g1_adv_loss
        
        if self.losses.get("sr_pix_trans"):
            # do the pixel-wise loss between syn_hr and syn_sr
            for p in self.net_g.parameters():
                p.requires_grad = False
            sr_pix = self.losses["sr_pix_trans"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_trans"] = sr_pix.item()
            loss_G += sr_pix
        
        if self.losses.get("noise_mean"):
            # add the mean of noise to make it close to zero: 
            # TODO: in OCT the noise mean might not be zero
            noise = self.predicted_noise
            noise_mean = (
                self.losses["noise_mean"](noise, torch.zeros_like(noise))
            )
            loss_dict["noise_mean"] = noise_mean.item()
            loss_G += noise_mean
        # Optimize the network_deg TODO:
        self.set_optimizer(names=["network_deg"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["network_deg"], self.max_grad_norm)
        self.set_optimizer(names=["network_deg"], operation="step")

        ## update D
        if step % self.D_ratio == 0:
            self.set_requires_grad(["network_dis1"], True)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            # Calculate the loss of real and fake on discriminating real and fake both
            loss_d1 = self.calculate_gan_loss_D(
                self.network_dis1, self.losses["lr_adv"],
                real, self.fake_lr_buffer.choose(fake)
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_D = loss_d1
            self.optimizers["network_dis1"].zero_grad()
            loss_D.backward()
            self.clip_grad_norm(["network_dis1"], self.max_grad_norm)
            self.optimizers["network_dis1"].step()
        
        

        return loss_dict
    
    def optimize_sr_models(self, step, loss_dict):
        # Only update net_g
        for p in self.net_g.parameters():
                p.requires_grad = True
        
        self.set_requires_grad(["network_deg"], False)
        self.sr_forward()
        loss_G = 0

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["network_dis2"], False)
            # calculate the loss of realness of syn_sr
            sr_adv_loss = self.calculate_gan_loss_G(
                self.network_dis2, self.losses["sr_adv"],
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_adv"] = sr_adv_loss.item()
            loss_G += sr_adv_loss
        
        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                loss_G += sr_style
            loss_G += sr_percep

        if self.losses.get("sr_pix_sr"):
            sr_pix = self.losses["sr_pix_sr"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_sr"] = sr_pix.item()
            loss_G += sr_pix

        self.set_optimizer(names=["network_g"], operation="zero_grad")
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.max_grad_norm)
        self.set_optimizer(names=["network_g"], operation="step")

        ## update D2
        if step % self.D_ratio == 0:
            if self.losses.get("sr_adv"):
                self.set_requires_grad(["network_dis2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.network_dis2, self.losses["sr_adv"],
                    self.syn_hr, self.fake_hr_buffer.choose(self.syn_sr)
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_D = loss_d2
                self.optimizers["network_dis2"].zero_grad()
                loss_D.backward()
                self.clip_grad_norm(["network_dis2"], self.max_grad_norm)
                self.optimizers["network_dis2"].step()
        
        return loss_dict


    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        # optimize trans
        if self.optim_deg:
            loss_dict = self.optimize_trans_models(step, loss_dict)

        # optimize SR
        if self.optim_sr:
            loss_dict = self.optimize_sr_models(step, loss_dict)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        if fake.size(1) == 3:
            d_pred_fake = netD(fake[:,1,:,:].unsqueeze(1).detach())
        else:
            d_pred_fake = netD(fake.detach())
        if real.size(1) == 3:
            d_pred_real = netD(real[:,1,:,:].unsqueeze(1))    
        else:
            d_pred_real = netD(real)

        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        # if fake.size(1) == 3:
        #     fake = fake.mean(dim=1, keepdim=True)
        if fake.size(1) == 3:
            d_pred_fake = netD(fake.mean(dim=1, keepdim=True))
        else:
            d_pred_fake = netD(fake)
        loss_real = criterion(d_pred_fake, True, is_disc=False)

        return loss_real
    
    def set_network_state(self, names, state):
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                getattr(self.networks[name], state)()

    def set_requires_grad(self, names, requires_grad):
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                for v in self.networks[name].parameters():
                    v.requires_grad = requires_grad

    def set_optimizer(self, names, operation):
        for name in names:
            getattr(self.optimizers[name], operation)()

    def clip_grad_norm(self, names, norm):
        for name in names:
            nn.utils.clip_grad_norm_(self.networks[name].parameters(), max_norm=norm)
    


    def test(self, test_data, crop_size=None):
        self.lq = test_data["lq"].to(self.device)
        if test_data.get("gt") is not None:
            self.gt = test_data["gt"].to(self.device)
            
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if crop_size is None:
                    self.fake_tgt = self.net_g(self.lq)
                    if self.fake_tgt.size(1) == 3:
                        self.fake_tgt = rgb_to_grayscale(self.fake_tgt)
                else:
                    self.fake_tgt = self.crop_test(self.lq, crop_size)
            self.net_g.train()

        if hasattr(self, "network_deg"):
            self.set_network_state(["network_deg"], "eval")
            if hasattr(self, "gt"):
                with torch.no_grad():
                    self.fake_lr = self.network_deg(self.gt)[0]
            self.set_network_state(["network_deg"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.lq.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
        if hasattr(self, "fake_lr"):
            out_dict["fake_lr"] = self.fake_lr.detach()[0].float().cpu()
        return out_dict
    
    def crop_test(self, lr, crop_size):
        b, c, h, w = lr.shape
        scale = self.opt["scale"]

        h_start = list(range(0, h-crop_size, crop_size))
        w_start = list(range(0, w-crop_size, crop_size))

        sr1 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hs in h_start:
            for ws in w_start:
                lr_patch = lr[:, :, hs: hs+crop_size, ws: ws+crop_size]
                sr_patch = self.net_g(lr_patch)
                if sr_patch.size(1) == 3:
                    sr_patch = rgb_to_grayscale(sr_patch)

                sr1[:, :, 
                    int(hs*scale):int((hs+crop_size)*scale),
                    int(ws*scale):int((ws+crop_size)*scale)
                ] = sr_patch
        
        h_end = list(range(h, crop_size, -crop_size))
        w_end = list(range(w, crop_size, -crop_size))

        sr2 = torch.zeros(b, c, int(h*scale), int(w* scale), device=self.device) - 1
        for hd in h_end:
            for wd in w_end:
                lr_patch = lr[:, :, hd-crop_size:hd, wd-crop_size:wd]
                sr_patch = self.net_g(lr_patch)
                if sr_patch.size(1) == 3:
                    sr_patch = rgb_to_grayscale(sr_patch)

                

                sr2[:, :, 
                    int((hd-crop_size)*scale):int(hd*scale),
                    int((wd-crop_size)*scale):int(wd*scale)
                ] = sr_patch

        mask1 = (
            (sr1 == -1).float() * 0 + 
            (sr2 == -1).float() * 1 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        mask2 = (
            (sr1 == -1).float() * 1 + 
            (sr2 == -1).float() * 0 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        sr = mask1 * sr1 + mask2 * sr2

        return sr


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return  images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images