import torch
import math

def feature_loss(fmap_r, fmap_g):
    """Pérdida de características con clamping para estabilidad"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach().clamp(-1e3, 1e3)
            gl = gl.float().clamp(-1e3, 1e3)
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 0.5  # Reducción de escala para mejor convergencia

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Pérdida del discriminador con regularización"""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float().clamp(1e-6, 1.0)
        dg = dg.float().clamp(1e-6, 1.0)
        loss += 0.5 * (torch.mean((1 - dr)**2) + torch.mean(dg**2))
    return loss

def generator_loss(disc_outputs):
    """Pérdida del generador con clamping"""
    loss = 0
    for dg in disc_outputs:
        dg = dg.float().clamp(1e-6, 0.999)
        loss += torch.mean((1 - dg)**2)
    return loss * 0.7  # Escalado para balancear con otras pérdidas

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """Pérdida KL con protección contra NaN"""
    z_p = z_p.float().clamp(-1e3, 1e3)
    logs_q = logs_q.float().clamp(-1e3, 1e3)
    m_p = m_p.float().clamp(-1e3, 1e3)
    logs_p = logs_p.float().clamp(-1e3, 1e3)
    
    kl = (logs_p - logs_q - 0.5 + 
          0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p))
    return torch.sum(kl * z_mask) / torch.sum(z_mask)