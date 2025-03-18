import contextlib
import importlib.metadata
import inspect
import json
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union
from urllib.parse import unquote

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml

# Basic environment info
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

# Global toggles
VERBOSE = True  # set False to reduce console output
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None

# Some OS booleans
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision

# -----------------------------------------------------------------------------
# CUSTOM PROGRESS BAR CLASS
# -----------------------------------------------------------------------------
class CustomTQDM(tqdm.tqdm):
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings
    and provides additional customization options.
    """

    def __init__(self, *args, **kwargs):
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)

# -----------------------------------------------------------------------------
# SIMPLE CLASS FOR PRETTY STRING REPRESENTATION
# -----------------------------------------------------------------------------
class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.
    """

    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                attr.append(f"{a}: {repr(v)}")
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'.")

# -----------------------------------------------------------------------------
# ITERABLE SIMPLE NAMESPACE
# -----------------------------------------------------------------------------
class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides iteration and custom attribute access.
    """

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'.")

    def get(self, key, default=None):
        return getattr(self, key, default)

# -----------------------------------------------------------------------------
# CUSTOM MATPLOTLIB SETTINGS DECORATOR
# -----------------------------------------------------------------------------
def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        def wrapper(*args, **kwargs):
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result
        return wrapper
    return decorator

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
def set_logging(name="GENERIC_LOGGER", verbose=True):
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR
    formatter = logging.Formatter("%(message)s")

    # On Windows, try to reconfigure stdout to UTF-8 if possible
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                return super().format(record)

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger

LOGGER = set_logging("GENERIC_LOGGER", verbose=VERBOSE)

# -----------------------------------------------------------------------------
# MISC HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def remove_color_escapes(input_string):
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)

def colorstr(*input):
    """
    Colors a string based on provided color/style arguments. 
    If you just pass one argument, it defaults to a certain style.
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m",
        "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
        "bright_black": "\033[90m", "bright_red": "\033[91m", "bright_green": "\033[92m", "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m", "bright_magenta": "\033[95m", "bright_cyan": "\033[96m", "bright_white": "\033[97m",
        "end": "\033[0m", "bold": "\033[1m", "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method.
    """

    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)
        return decorated

def yaml_save(file="data.yaml", data=None, header=""):
    """
    Save YAML data to a file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    with open(file, "w", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.
    """
    file = Path(file)
    assert file.suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file}"
    with open(file, encoding="utf-8") as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

# -----------------------------------------------------------------------------
# SIMPLE JSON PERSISTENCE
# -----------------------------------------------------------------------------
class JSONDict(dict):
    """
    A dictionary-like class that provides JSON persistence for its contents. Thread-safe using a lock.
    """

    def __init__(self, file_path: Union[str, Path] = "data.json"):
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        try:
            if self.file_path.exists():
                with open(self.file_path, encoding="utf-8") as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}. Starting empty.")
        except Exception as e:
            print(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            print(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)
            self._save()

    def update(self, *args, **kwargs):
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        with self.lock:
            super().clear()
            self._save()

    def __str__(self):
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

# -----------------------------------------------------------------------------
# SIMPLE TRY/EXCEPT DECORATOR
# -----------------------------------------------------------------------------
class TryExcept(contextlib.ContextDecorator):
    """
    A context manager and decorator for gracefully handling exceptions with an optional message.
    """

    def __init__(self, msg="", verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True  # suppresses the exception

# -----------------------------------------------------------------------------
# SIMPLE RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# -----------------------------------------------------------------------------
class Retry(contextlib.ContextDecorator):
    """
    A decorator/context manager for retrying a function with exponential backoff.
    """

    def __init__(self, times=3, delay=2):
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))
        return wrapped_func

# -----------------------------------------------------------------------------
# THREADED DECORATOR
# -----------------------------------------------------------------------------
def threaded(func):
    """
    Decorator that runs a function in a separate thread by default. 
    If 'threaded=False' is passed, it runs in the main thread.
    """

    def wrapper(*args, **kwargs):
        if kwargs.pop("threaded", True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)
    return wrapper

# End of brand-neutral utility code.
