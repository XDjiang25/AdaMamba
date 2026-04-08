import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Embed import MultiScalePatchEmbedding
import math
import torch.fft as fft  


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.revin = configs.revin
        if configs.revin==1:
            self.revin_layer = RevIN(configs.enc_in)
        # patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        if isinstance(configs.patch_lens, str):
            self.patch_lens = [int(p) for p in configs.patch_lens.split(',')]
        else:
            self.patch_lens = configs.patch_lens

        self.multiscale_patchembedding = MultiScalePatchEmbedding(configs.d_model, patch_lens=self.patch_lens, stride=self.stride, dropout=configs.dropout)
        self.patch_nums = sum([int((configs.seq_len - patch_len) / self.stride + 2) for patch_len in self.patch_lens])
        

        self.encoder = SFMMEnocder(configs)
        self.head_nf = configs.d_model * self.patch_nums
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                        head_dropout=configs.head_dropout)
        self.proj = nn.Linear(configs.dim_pitch, configs.d_model)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.ones([1]) * configs.beta)
        self.gama = nn.Parameter(torch.tensor(1.0))
      
        self.correlation_embedding = nn.Conv1d(configs.enc_in, configs.enc_in, configs.ckernel, padding='same')
        self.dropoutlayer = nn.Dropout(configs.dropout)


    def forward(self, x_enc, return_A=False):

        if self.revin==1:
            x_enc = self.revin_layer(x_enc,'norm')
        else:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, T, N = x_enc.shape # B L N
        x_enc = x_enc.permute(0, 2, 1) # B N L

        enc_out = self.correlation_embedding(x_enc)
        enc_ts = self.beta * x_enc + (1 - self.beta) * enc_out
        enc_out, n_vars = self.multiscale_patchembedding(enc_ts)  # [bs*nvars, patch_num, d_model] enc_out.shape torch.Size([360, 171, 16])
        if return_A:

            enc_out_dd, A_final, omega_final, omega_offsets, g_final = self.encoder(enc_out, return_A=True)
        else:
            enc_out_dd = self.encoder(enc_out, return_A=False)# dim_pitch == d_model

        
        enc_out = torch.reshape(enc_out_dd, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out[:, :, :, -self.patch_nums:])  # dec_out: [bs, nvars, target_window]

        dec_out = dec_out.permute(0, 2, 1)

        
        if self.revin ==1:
            dec_out = self.revin_layer(dec_out,'denorm')

        else:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        if return_A:
            return dec_out[:, -self.pred_len:, :], dec_out[:, -self.pred_len:, :], A_final, omega_final, omega_offsets, g_final
        
        return dec_out[:, -self.pred_len:, :], dec_out[:, -self.pred_len:, :]            


def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

    
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
                    
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
        # prepare tensors
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply

class SFMMEnocder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.dropout = nn.Dropout(configs.dropout)
        self.layers = nn.ModuleList([SFMMBlock(configs) for _ in range(configs.e_layers)])
       

    def forward(self, x, return_A=False):
        # x : [bs * nvars, patch_num, d_model]
        
        for i, layer in enumerate(self.layers):
            
            x_in = x
        
            
            is_last_layer = (i == len(self.layers) - 1)

            if return_A and is_last_layer:

                x_out, A_final, omega_final, omega_offsets, g_final = layer(x_in, return_A=True)
            else:

                x_out = layer(x_in, return_A=False)
            
            x = self.dropout(x_out + x_in)
        if return_A:
            
            return x, A_final, omega_final, omega_offsets, g_final
            
        return x    

class SFMMBlock(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        dim = configs.d_model
        self.dim_feq = configs.dim_feq
        dim_pitch = configs.dim_pitch
        assert dim == dim_pitch, "For z dependency with teacher forcing, dim must equal dim_pitch"
        bias = getattr(configs, 'bias', True)
        
        
        
        self.gating_type = getattr(configs, 'gating_type', 'sigmoid') 
        print(f"Using gating activation: {self.gating_type}")

        # Replace fixed omega
        self.omega_base = nn.Parameter(2 * torch.pi * torch.arange(self.dim_feq) / self.dim_feq)

        # Add a small MLP to adapt omega per batch/sequence
        self.omega_adapter = nn.Sequential(
            nn.Linear(dim, dim // 4),  # Pool input to low-dim
            nn.ReLU(),
            nn.Linear(dim // 4, self.dim_feq)  # Output offsets for omega
        )        
        # Projections for gates with z dependency (W_*) and x (V_*)
        self.W_ste = nn.Linear(dim_pitch, self.dim_feq, bias=bias)
        self.V_ste = nn.Linear(dim, self.dim_feq, bias=bias)

        self.W_fre = nn.Linear(dim_pitch, dim, bias=bias)
        self.V_fre = nn.Linear(dim, dim, bias=bias)

        # Update W_g, V_g, etc., to output dim_feq channels
        self.W_g = nn.Linear(dim_pitch, dim * self.dim_feq, bias=bias)
        self.V_g = nn.Linear(dim, dim * self.dim_feq, bias=bias)
        self.W_i = nn.Linear(dim_pitch, dim * self.dim_feq, bias=bias)
        self.V_i = nn.Linear(dim, dim * self.dim_feq, bias=bias)

        # # FEQ params (ignoring W_o for z in o, as per approximation)
        self.U_o = nn.Parameter(torch.randn(self.dim_feq, dim)) # SxN
        self.U_o_phi = nn.Parameter(torch.randn(self.dim_feq, dim)) # SxN

        self.V_o = nn.Parameter(torch.randn(self.dim_feq, dim))
        self.W_o = nn.Linear(dim_pitch, self.dim_feq, bias=bias)

        self.b_o = nn.Parameter(torch.zeros(self.dim_feq))
        self.W_z = nn.Parameter(torch.randn(self.dim_feq, dim, dim_pitch))
        self.b_z = nn.Parameter(torch.zeros(self.dim_feq, dim_pitch))

        # Learnable freq weights
        self.freq_weights = nn.Parameter(torch.ones(self.dim_feq) / self.dim_feq)  # Normalize init

        # Optional dynamic: MLP from pooled input
        self.freq_attn = nn.Sequential(
            nn.Linear(dim, self.dim_feq),
            nn.Softmax(dim=-1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = 0.1
        for name, param in self.named_parameters():
            if 'weight' in name.lower():
                if param.dim() >= 2:  
                    if isinstance(param, nn.Linear):
                        nn.init.xavier_uniform_(param.weight, gain=gain)
                    else:
                        nn.init.xavier_uniform_(param, gain=gain)
                else:  
                    nn.init.uniform_(param, a=-0.1, b=0.1)  
            elif 'bias' in name.lower():
                nn.init.zeros_(param)

    
    def get_gating_activation(self, x):
        if self.gating_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.gating_type == 'tanh':
            return torch.tanh(x)
        elif self.gating_type == 'silu':
            return F.silu(x)
        else:
            return torch.sigmoid(x) 


    def forward(self, x, return_A=False):
            # x: [bs, nsteps, dim]
            bs, nsteps, dim = x.shape
            device = x.device

            
            proxy_z = torch.roll(x, shifts=1, dims=1)  # [bs, nsteps, dim]
            proxy_z[:, 0, :] = 0.0  # Initial z = 0

            # Normalized times
            times = torch.arange(1, nsteps + 1, device=device).float() / nsteps  # [nsteps]

            
            seq_pool = x.mean(dim=1)  # [bs, dim]
            omega_offsets = self.omega_adapter(seq_pool)  # [bs, dim_feq]
            # learnable
            
            omega = (self.omega_base[None, :] + omega_offsets).clamp(min=0.0)  # [bs, dim_feq]
            if return_A:
                
                base_vals = [round(val.item(), 3) for val in self.omega_base[:10]]
                print(f"omega_base: {base_vals}")
                
                
                final_vals = [round(val.item(), 3) for val in omega[0, :10]]
                print(f"omega_final: {final_vals}")   
            

            cos_t = torch.cos(torch.einsum('b k, n -> b n k', omega, times))  # [bs, nsteps, dim_feq]
            sin_t = torch.sin(torch.einsum('b k, n -> b n k', omega, times))  # [bs, nsteps, dim_feq]

            # Compute gates with z dependency
            f_ste = torch.sigmoid(self.W_ste(proxy_z) + self.V_ste(x))  # [bs, nsteps, dim_feq]
            f_fre = torch.sigmoid(self.W_fre(proxy_z) + self.V_fre(x))  # [bs, nsteps, dim]
            f = f_fre[:, :, :, None] * f_ste[:, :, None, :]  # [bs, nsteps, dim, dim_feq]


            g_raw = self.W_g(proxy_z) + self.V_g(x)  # [bs, nsteps, dim * dim_feq]

            g = self.get_gating_activation(g_raw.view(bs, nsteps, dim, self.dim_feq))  # [bs, nsteps, dim, dim_feq]
            # g = torch.sigmoid(g_raw.view(bs, nsteps, dim, self.dim_feq))  # [bs, nsteps, dim, dim_feq]

            i_raw = self.W_i(proxy_z) + self.V_i(x)  # [bs, nsteps, dim * dim_feq]
            i = torch.tanh(i_raw.view(bs, nsteps, dim, self.dim_feq))  # [bs, nsteps, dim, dim_feq]
            gi = g * i  # [bs, nsteps, dim, dim_feq]

            # Modulations with per-frequency gi
            gi_cos = gi * cos_t[:, :, None, :]  # [bs, nsteps, dim, dim_feq]
            gi_sin = gi * sin_t[:, :, None, :]  # [bs, nsteps, dim, dim_feq]

            # States via pscan
            Re_s = pscan(f, gi_cos)  # [bs, nsteps, dim, dim_feq]
            Im_s = pscan(f, gi_sin)  # [bs, nsteps, dim, dim_feq]

            # Amplitude
            A = torch.sqrt(Re_s**2 + Im_s**2 + 1e-8)  # [bs, nsteps, dim, dim_feq]
            A_trans = A.permute(0, 1, 3, 2)  # [bs, nsteps, dim_feq, dim]
            # phase
            phi = torch.atan2(Im_s, Re_s + 1e-8)  # [bs, nsteps, dim, dim_feq]
            phi_trans = phi.permute(0, 1, 3, 2)  # [bs, nsteps, dim_feq, dim]

            # FEQ
            u_term_amp = torch.einsum('blfd,fd->blf', A_trans, self.U_o)  # [bs, nsteps, dim_feq]
            u_term_phi = torch.einsum('blfd,fd->blf', phi_trans, self.U_o_phi)

            v_term = torch.einsum('bld,fd->blf', x, self.V_o)  # [bs, nsteps, dim_feq]
            w_term = self.W_o(proxy_z)
            o = torch.sigmoid(u_term_amp + v_term + w_term + self.b_o[None, None, :])  # [bs, nsteps, dim_feq]
            


            tanh_input = torch.einsum('blfd,fpd->blfp', A_trans, self.W_z) + self.b_z[None, None, :, :]  # [bs, nsteps, dim_feq, dim_pitch]
            tanh_term = torch.tanh(tanh_input)

            delta = o.unsqueeze(3) * tanh_term  # [bs, nsteps, dim_feq, dim_pitch]

            z = delta.sum(dim=2)  # [bs, nsteps, dim_pitch]

            if return_A:

                return z, A, omega, omega_offsets, g
            return z




