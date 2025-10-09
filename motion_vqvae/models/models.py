import torch
import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset

from torch.utils.data import TensorDataset, DataLoader

class VQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.config['quantizer']
        self.encoder = Encoder(args.config['frame_size'], output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(args.config['frame_size'], output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.config['quantizer'] == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.config['quantizer'] == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.config['quantizer'] == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.config['quantizer'] == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

    @torch.no_grad()
    def get_codebook_seq(self, 
                         mocap_data: torch.Tensor, 
                         end_indices: list, 
                         window_size: int, 
                         mean: torch.Tensor, 
                         std: torch.Tensor, 
                         batch_size: int = 256,
                         device: torch.device = torch.device("cpu")):
        """
        전체 모션 데이터셋을 '클립 단위'로 분할하고,
        각 클립 내에서만 슬라이딩 윈도우를 적용하여 잠재 시퀀스를 추출합니다.
        """
        self.eval()
        device = next(self.parameters()).device

        # 1. Split mocap_data into individual motion clips
        motion_clips = []
        start_idx = 0
        for end_idx in end_indices:
            motion_clips.append(mocap_data[start_idx : end_idx + 1])
            start_idx = end_idx + 1

        # 2. Extract all valid windows within each clip
        all_windows = []
        # Metadata for referencing the original motion in the DFM dataset
        # (clip index, start frame within clip)
        all_windows_metadata = [] 
        
        for clip_idx, clip in enumerate(motion_clips):
            # Apply sliding window only if the current clip is longer than window_size
            if len(clip) >= window_size:
                for i in range(len(clip) - window_size + 1):
                    all_windows.append(clip[i : i + window_size])
                    all_windows_metadata.append((clip_idx, i))
            print(f"clip_idx: {clip_idx}, clip: {clip.shape}, collected windows: {len(all_windows), i}")

        if not all_windows:
            return torch.tensor([]), [] # No valid windows

        window_tensor = torch.stack(all_windows, dim=0)
        normalized_windows = (window_tensor - mean) / std
        
        # 3. Efficient encoding by batching through DataLoader
        dataset = TensorDataset(normalized_windows)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_code_indices = []
        for batch in loader:
            motion_window_batch = batch[0].to(device)
            code_indices = self.encode(motion_window_batch)
            all_code_indices.append(code_indices.cpu())
            
        return torch.cat(all_code_indices, dim=0), motion_clips, all_windows_metadata



class MotionVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = args.config['num_joints']
        self.vqvae = VQVAE(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
    
    def get_codebook_seq(self, 
                         mocap_data: torch.Tensor, 
                         end_indices: list, 
                         window_size: int, 
                         mean: torch.Tensor, 
                         std: torch.Tensor, 
                         batch_size: int = 256,
                         device: torch.device = torch.device("cpu")):
        """Get codebook sequences from the model."""
        return self.vqvae.get_codebook_seq(mocap_data, end_indices, window_size, mean, std, batch_size, device)