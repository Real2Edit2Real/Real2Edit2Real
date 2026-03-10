import time
import math
from collections.abc import Mapping
from addict import Dict as Addict


class ProgressTracker:
    """
    A class to track iteration progress and provide formatted strings
    without using the tqdm library, with dynamic rate formatting.
    """
    def __init__(self, total_iterations, description=""):
        """
        Initializes the progress tracker.

        Args:
            total_iterations (int): The total number of iterations to track.
            description (str, optional): A description for the process. Defaults to "".
        """
        if not isinstance(total_iterations, int) or total_iterations <= 0:
            raise ValueError("total_iterations must be a positive integer.")

        self.total_iterations = total_iterations
        self.description = description
        self.start_time = None
        self.current_iteration = 0
        self.last_update_time = None
        self.iteration_times = [] # To store a history for smoother rate calculation

        # Default values for initial state
        self._elapsed_time = 0.0
        self._rate_per_second = 0.0 # Raw rate in iterations per second
        self._remaining_time = float('inf')

    def start(self):
        """
        Starts the progress tracking. Call this before the loop begins.
        """
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self.current_iteration = 0
        self.iteration_times = []

    def update(self, count=1):
        """
        Updates the progress by the given count. Call this inside your loop.

        Args:
            count (int): The number of iterations completed since the last update. Defaults to 1.
        """
        if self.start_time is None:
            raise RuntimeError("Call .start() before calling .update()")

        self.current_iteration += count
        current_time = time.monotonic()
        
        # Calculate iteration time for rate
        if self.current_iteration > 0: # Only start calculating after first iteration
            time_since_last_update = current_time - self.last_update_time  # pyright: ignore
            # Store average time per item for this update
            self.iteration_times.append(time_since_last_update / count)
            
            # Keep a reasonable history (e.g., last 100 updates for average rate)
            if len(self.iteration_times) > 100:
                self.iteration_times.pop(0)

        self.last_update_time = current_time

        self._calculate_metrics()

    def _calculate_metrics(self):
        """
        Internal method to calculate elapsed time, rate, and remaining time.
        """
        current_time = time.monotonic()
        self._elapsed_time = current_time - self.start_time  # pyright: ignore

        if self.current_iteration > 0 and len(self.iteration_times) > 0:
            average_item_time = sum(self.iteration_times) / len(self.iteration_times)
            self._rate_per_second = 1.0 / average_item_time if average_item_time > 0 else 0.0
            
            remaining_iterations = self.total_iterations - self.current_iteration
            if remaining_iterations > 0 and self._rate_per_second > 0:
                self._remaining_time = remaining_iterations / self._rate_per_second
            else:
                self._remaining_time = 0.0 # All done or rate is zero
        else:
            self._rate_per_second = 0.0
            self._remaining_time = float('inf') # Still at the beginning

    def _format_rate(self):
        """
        Dynamically formats the iteration rate (it/s, s/it, min/it).
        """
        if self._rate_per_second == 0 and self.current_iteration == 0:
            return "---it/s" # Before any work is done
        
        if self._rate_per_second >= 1.0: # Faster than 1 it/s
            return f"{self._rate_per_second:.2f}it/s"
        elif self._rate_per_second > 0: # Slower than 1 it/s, but not extremely slow
            seconds_per_iteration = 1.0 / self._rate_per_second
            if seconds_per_iteration < 60: # Less than 60 seconds per iteration
                return f"{seconds_per_iteration:.2f}s/it"
            else: # 60 seconds or more per iteration
                minutes_per_iteration = seconds_per_iteration / 60.0
                return f"{minutes_per_iteration:.2f}min/it"
        else: # Rate is effectively zero (e.g., no updates or very long gaps)
            return "0.00it/s"


    def get_progress_string(self):
        """
        Returns the current progress as a formatted string
        (e.g., '30/30 [00:09<00:00, 3.05it/s]').
        """
        n_fmt = f"{self.current_iteration}"
        total_fmt = f"{self.total_iterations}"

        elapsed_str = self._format_time(self._elapsed_time)
        
        # Handle infinite remaining time at the start
        if math.isinf(self._remaining_time):
            remaining_str = "?:??"
        else:
            remaining_str = self._format_time(self._remaining_time)

        rate_fmt = self._format_rate() # Use the new dynamic formatter

        return f"[{self.description}]: {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str}, {rate_fmt}]"

    def _format_time(self, seconds):
        """Helper to format time in HH:MM:SS or MM:SS."""
        if seconds < 0:
            seconds = 0
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def is_finished(self):
        """Returns True if all iterations are complete."""
        return self.current_iteration >= self.total_iterations

    def __str__(self):
        return self.get_progress_string()


def update_nested_dict(original_dict, new_kvs):
    """
    Recursively updates a nested dictionary with new key-value pairs.
    """
    for k, v in new_kvs.items():
        if isinstance(v, Mapping) and k in original_dict and isinstance(original_dict[k], Mapping):
            # If both are dicts, recurse
            original_dict[k] = update_nested_dict(original_dict[k], v)
        else:
            # Otherwise, overwrite
            original_dict[k] = v
    return original_dict


def unflatten_dict(d: dict, sep: str = '.'):
    """
    Unflattens a single-level dictionary with dot-separated keys back into a nested one.
    e.g., {'a.b': 1} -> {'a': {'b': 1}}
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_nested = result
        for part in parts[:-1]:
            if part not in d_nested:
                d_nested[part] = {}
            d_nested = d_nested[part]
        d_nested[parts[-1]] = value
    return result


class Dict(Addict):

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'") from e
