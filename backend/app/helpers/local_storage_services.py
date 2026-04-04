import os
import shutil
from dotenv import load_dotenv

load_dotenv()


def _human_readable_size(size_in_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"


def _entry_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


class LocalStorageManager:
    """Flat local filesystem storage under LOCAL_STORAGE_DIR."""

    def __init__(self):
        self.root = os.path.abspath(os.getenv("LOCAL_STORAGE_DIR", "./storage"))
        os.makedirs(self.root, exist_ok=True)

    def _resolve_under_root(self, rel: str) -> str:
        if not rel or not str(rel).strip():
            raise ValueError("Invalid storage path")
        base = os.path.abspath(self.root)
        full = os.path.normpath(os.path.join(base, rel))
        if not full.startswith(base + os.sep) and full != base:
            raise ValueError("Path escapes storage root")
        return full

    def get_path(self, name: str) -> str:
        return self._resolve_under_root(name)

    def path_exists(self, name: str) -> bool:
        try:
            return os.path.exists(self._resolve_under_root(name))
        except ValueError:
            return False

    async def delete_file(self, filename: str) -> None:
        path = self._resolve_under_root(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {filename}")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    async def rename_file_or_folder(
        self, old_name: str, new_name: str, ignore_missing: bool = False
    ) -> None:
        src = self.get_path(old_name)
        dst = self.get_path(new_name)
        if not os.path.exists(src):
            if ignore_missing:
                return
            raise FileNotFoundError(f"Not found: {old_name}")
        os.makedirs(os.path.dirname(dst) or self.root, exist_ok=True)
        shutil.move(src, dst)

    def save_file(self, source_path: str, filename: str) -> None:
        dst = self.get_path(filename)
        os.makedirs(os.path.dirname(dst) or self.root, exist_ok=True)
        shutil.copy2(source_path, dst)

    async def list_recent_uploads(self):
        label = os.getenv("RECENTLY_UPLOADED_DATASETS_DIR", "uploads")
        formatted = []
        try:
            for name in sorted(os.listdir(self.root)):
                path = os.path.join(self.root, name)
                if name.startswith(".") or not os.path.exists(path):
                    continue
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    ftype = "FILE"
                elif os.path.isdir(path):
                    size = _entry_size(path)
                    ftype = "DIRECTORY"
                else:
                    continue
                formatted.append(
                    {
                        "filename": name,
                        "size": _human_readable_size(size),
                        "type": ftype,
                    }
                )
            return {"contents": {label: formatted}, "error": None}
        except Exception as e:
            print(f"Error listing storage: {e}")
            raise

    def list_files_upload_response(self) -> dict:
        """Shape compatible with legacy /file-upload/list-files."""
        formatted = []
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if name.startswith("."):
                continue
            if os.path.isfile(path):
                formatted.append(
                    {
                        "filename": name,
                        "size": os.path.getsize(path),
                        "type": "FILE",
                        "modification_time": int(os.path.getmtime(path) * 1000),
                        "permission": None,
                    }
                )
            elif os.path.isdir(path):
                formatted.append(
                    {
                        "filename": name,
                        "size": _entry_size(path),
                        "type": "DIRECTORY",
                        "modification_time": int(os.path.getmtime(path) * 1000),
                        "permission": None,
                    }
                )
        return {"contents": {"uploads": formatted}, "error": None}

    def copy_storage_entry_to_local(self, filename: str, local_destination_path: str) -> None:
        """Copy a file or parquet directory from storage root into a local folder."""
        src = self.get_path(filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Storage entry not found: {filename}")
        os.makedirs(local_destination_path, exist_ok=True)
        dst = os.path.join(local_destination_path, os.path.basename(src))
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def check_file_exists(self, name: str) -> bool:
        return self.path_exists(name)
