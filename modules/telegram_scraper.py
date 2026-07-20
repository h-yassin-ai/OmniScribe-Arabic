import os
import asyncio
from telethon import TelegramClient
from telethon.tl.types import MessageMediaDocument, DocumentAttributeFilename
from modules.utils import logger, DOWNLOADS_DIR

client = None

def get_client(api_id, api_hash, session_name="kutub_telegram_session"):
    global client
    # Place session file in outputs or root directory safely
    session_path = os.path.join(os.path.dirname(DOWNLOADS_DIR), session_name)
    if client is None:
        client = TelegramClient(session_path, int(api_id), api_hash)
    return client

async def send_telegram_code(api_id, api_hash, phone):
    global client
    client = get_client(api_id, api_hash)
    if not client.is_connected():
        await client.connect()
    
    # Send code request
    sent = await client.send_code_request(phone)
    return sent.phone_code_hash

async def sign_in_telegram(phone, phone_code_hash, code):
    global client
    if client is None:
        return False, "Client not initialized"
    if not client.is_connected():
        await client.connect()
    try:
        await client.sign_in(phone, code, phone_code_hash=phone_code_hash)
        return True, "Success"
    except Exception as e:
        return False, str(e)

async def is_telegram_authorized(api_id=None, api_hash=None):
    global client
    if client is None:
        if api_id and api_hash:
            get_client(api_id, api_hash)
        else:
            from modules.utils import BASE_DIR
            env_path = os.path.join(BASE_DIR, ".env")
            api_id_val = None
            api_hash_val = None
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "=" in line and not line.strip().startswith("#"):
                                k, v = line.split("=", 1)
                                if k.strip() == "TG_API_ID":
                                    api_id_val = v.strip()
                                elif k.strip() == "TG_API_HASH":
                                    api_hash_val = v.strip()
                except Exception:
                    pass
            if api_id_val and api_hash_val:
                try:
                    get_client(api_id_val, api_hash_val)
                except Exception:
                    return False
            else:
                return False
    if not client.is_connected():
        try:
            await client.connect()
        except Exception:
            return False
    try:
        return await client.is_user_authorized()
    except Exception:
        return False
_workspace_files_cache = None

def get_workspace_files_cache():
    global _workspace_files_cache
    if _workspace_files_cache is not None:
        return _workspace_files_cache
        
    from modules.utils import BASE_DIR
    cache = {}
    excluded_dirs = {'.git', '.venv', 'node_modules', '__pycache__', 'models', 'modules', '.agents', '.gemini'}
    
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for f in files:
            full_path = os.path.join(root, f)
            try:
                size = os.path.getsize(full_path)
                cache[(f, size)] = full_path
            except OSError:
                continue
                
    _workspace_files_cache = cache
    return _workspace_files_cache

def clear_workspace_files_cache():
    global _workspace_files_cache
    _workspace_files_cache = None

async def list_channel_files(channel_name, limit=100):
    global client
    if client is None or not await is_telegram_authorized():
        raise ConnectionError("Telegram client is not authorized. Please log in first.")
        
    entity = await client.get_input_entity(channel_name)
    files = []
    
    # Refresh cache for files list
    clear_workspace_files_cache()
    cache = get_workspace_files_cache()
    
    async for msg in client.iter_messages(entity, limit=limit):
        if msg.media and isinstance(msg.media, MessageMediaDocument):
            doc = msg.media.document
            filename = "file_without_name"
            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeFilename):
                    filename = attr.file_name
                    break
            
            # Check if file has audio/video extension or mime-type
            mime = doc.mime_type or ""
            if any(ext in filename.lower() or ext in mime.lower() for ext in [".m4a", ".mp3", ".wav", ".mp4", ".mkv", ".ogg", "audio", "video"]):
                size_mb = doc.size / (1024 * 1024)
                
                if filename.startswith('.'):
                    clean_filename = "file_" + filename.lstrip('.')
                    if not clean_filename or clean_filename == "file_":
                        clean_filename = "file_unnamed"
                else:
                    clean_filename = filename
                exists = (clean_filename, doc.size) in cache
                
                files.append({
                    "id": msg.id,
                    "name": filename,
                    "size": f"{size_mb:.2f} MB",
                    "msg_id": msg.id,
                    "date": msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                    "exists": exists
                })
    return files

async def download_telegram_file(channel_name, msg_id, progress_callback=None, dest_dir=None):
    global client
    if client is None or not await is_telegram_authorized():
        raise ConnectionError("Telegram client is not authorized.")
        
    entity = await client.get_input_entity(channel_name)
    msg = await client.get_messages(entity, ids=msg_id)
    
    if not msg or not msg.media:
        raise ValueError(f"No media found in message ID {msg_id}")
        
    # Get filename
    filename = "tg_file"
    doc = msg.media.document
    for attr in doc.attributes:
        if isinstance(attr, DocumentAttributeFilename):
            filename = attr.file_name
            break
            
    # Check cache first
    cache = get_workspace_files_cache()
    if filename.startswith('.'):
        clean_filename = "file_" + filename.lstrip('.')
        if not clean_filename or clean_filename == "file_":
            clean_filename = "file_unnamed"
    else:
        clean_filename = filename
        
    if (clean_filename, doc.size) in cache:
        existing_path = cache[(clean_filename, doc.size)]
        logger.info(f"File '{clean_filename}' already exists at: {existing_path}. Skipping download.")
        return existing_path, False
            
    target_dir = dest_dir if dest_dir is not None else DOWNLOADS_DIR
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, filename)
    
    # Download with callback
    def prog_call(current, total):
        if progress_callback:
            progress_callback(current, total)
            
    path = await client.download_media(msg, file=out_path, progress_callback=prog_call)
    
    from modules.utils import sanitize_path_prevent_dotfile
    path = sanitize_path_prevent_dotfile(path)
    
    # Update cache
    if _workspace_files_cache is not None:
        try:
            size = os.path.getsize(path)
            _workspace_files_cache[(os.path.basename(path), size)] = path
        except OSError:
            pass
            
    return path, True
