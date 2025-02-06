import os
import json
import logging
import torch

# Configuración inicial del logger
logger = logging.getLogger(__name__)

def setup_logger():
    """Configura el sistema de logging para Colab"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formato personalizado con emojis
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para Colab
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Limpiar handlers existentes y añadir nuevo
    logger.handlers = []
    logger.addHandler(handler)
    
    return logger

def get_hparams():
    """Carga y ajusta hiperparámetros con validación mejorada"""
    config_path = "/content/drive/MyDrive/vits/configs/ljs_base.json"
    
    try:
        with open(config_path, 'r') as f:
            hparams = json.load(f)
            
        # Ajustes automáticos para diferentes GPUs
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
        hparams['train']['batch_size'] = 4 if 'T4' in gpu_name else 8
        
        # Validación de parámetros esenciales
        required_keys = ['data', 'train', 'model']
        for key in required_keys:
            if key not in hparams:
                raise KeyError(f"Falta sección crítica en config: {key}")
                
        return hparams
        
    except FileNotFoundError:
        logger.critical(f"🚨 Archivo de configuración no encontrado: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.critical("🚨 Error de formato en el archivo de configuración")
        raise

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Carga segura de checkpoints con manejo de errores mejorado"""
    try:
        # Carga segura con weights_only=True
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=True
        )
        
        # Cargar estados
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"♻️ Checkpoint cargado: {os.path.basename(checkpoint_path)}")
        return model, optimizer, epoch
        
    except FileNotFoundError:
        logger.warning(f"⚠️ Checkpoint no encontrado: {checkpoint_path}")
        return model, optimizer, 0
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar checkpoint: {str(e)}")
        raise

def save_checkpoint(model, optimizer, epoch, path):
    """Guardado robusto de checkpoints con creación de directorios"""
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar con comprobación de seguridad
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
        
        logger.info(f"💾 Checkpoint guardado exitosamente: {os.path.basename(path)}")
        
    except Exception as e:
        logger.error(f"🔥 Error catastrófico al guardar checkpoint: {str(e)}")
        raise