import os
import sys
import json
import traceback
import re
import shutil
import contextlib
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


# Context manager to redirect stdout/stderr temporarily
class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or open(os.devnull, 'w')
        self._stderr = stderr or open(os.devnull, 'w')
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        if self._stdout != sys.stdout:
            self._stdout.close()
        if self._stderr != sys.stderr:
            self._stderr.close()


class BFCLEvaluator:
    """
    A minimal BFCL evaluator that directly uses the multi_turn_runner function
    to get scores without unnecessary code or file operations.
    
    This version reduces logging noise during evaluation when verbose=False.
    """
    
    def __init__(self, 
                 bfcl_path: Optional[str] = None, 
                 verbose: bool = False,
                 project_root: Optional[str] = None):
        """
        Initialize the minimal BFCL evaluator.
        
        Args:
            bfcl_path: Optional path to the BFCL module directory
            verbose: Whether to print verbose output and allow BFCL logs
            project_root: Root directory of the BFCL project (default: auto-detect)
        """
        self.verbose = verbose
        
        if bfcl_path and bfcl_path not in sys.path:
            sys.path.append(bfcl_path)
        
        # Import only the essential BFCL modules
        try:
            # Silence import noise if not verbose
            with self._silence_output():
                from bfcl.constants.category_mapping import VERSION_PREFIX
                from bfcl.constants.eval_config import (
                    PROMPT_PATH, PROJECT_ROOT, POSSIBLE_ANSWER_PATH,
                    RESULT_PATH, SCORE_PATH
                )
                from bfcl.utils import (
                    load_file, find_file_with_suffix, is_multi_turn
                )
                
                # Store the minimum required attributes
                self.VERSION_PREFIX = VERSION_PREFIX
                self.PROMPT_PATH = PROMPT_PATH
                self.POSSIBLE_ANSWER_PATH = POSSIBLE_ANSWER_PATH
                
                # Allow custom project root or use the one from BFCL
                self.PROJECT_ROOT = project_root or PROJECT_ROOT
                
                # Set up standard BFCL directories
                self.result_dir = RESULT_PATH if hasattr(RESULT_PATH, 'exists') else Path(os.path.join(self.PROJECT_ROOT, "result"))
                self.score_dir = SCORE_PATH if hasattr(SCORE_PATH, 'exists') else Path(os.path.join(self.PROJECT_ROOT, "score"))
                
                # Import only the needed utility functions
                self.load_file = load_file
                self.find_file_with_suffix = find_file_with_suffix
                self.is_multi_turn = is_multi_turn
                
                # Import just the multi_turn_runner and get_handler functions
                from bfcl.eval_checker.eval_runner import (
                    multi_turn_runner, get_handler
                )
                
                self.multi_turn_runner = multi_turn_runner
                self.get_handler = get_handler
                
                if self.verbose:
                    print(f"[INFO] Successfully imported BFCL modules")
                    
        except ImportError as e:
            print(f"[WARNING] Could not import BFCL modules: {e}")
            print(f"[WARNING] Make sure BFCL is installed or the path is correct")
            raise ImportError(f"Required BFCL modules not found: {str(e)}")
    
    def _silence_output(self):
        """
        Context manager to silence stdout/stderr.
        Only silences if verbose=False, otherwise allows output.
        """
        if not self.verbose:
            return RedirectStdStreams()
        else:
            # Return a dummy context manager that does nothing
            @contextlib.contextmanager
            def no_op():
                yield
            return no_op()
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages based on verbosity setting."""
        if self.verbose or level in ["WARNING", "ERROR"]:
            try:
                # Use bittensor logging if available
                import bittensor as bt
                if level == "INFO":
                    bt.logging.info(f"OFFLINE: {message}")
                elif level == "WARNING":
                    bt.logging.warning(f"OFFLINE: {message}")
                elif level == "ERROR":
                    bt.logging.error(f"OFFLINE: {message}")
                elif level == "DEBUG" and self.verbose:
                    bt.logging.debug(f"OFFLINE: {message}")
            except ImportError:
                # Fall back to print
                print(f"[{level}] OFFLINE: {message}")
    
    def evaluate_model(
        self, 
        model_name: str = "BitAgent",
        endpoint: str = "localhost", 
        port: int = 51001,
        test_category: str = "multi_turn_base",
        temperature: float = 0.001,
        backend: str = "sglang",
        cleanup_files: bool = True
    ) -> Dict[str, Any]:
        
        """
        Evaluate a model using BFCL's multi_turn_runner directly.
        
        Args:
            model_name: Name to use for the model in BFCL
            endpoint: Host where the model is being served
            port: Port where the model is being served
            test_category: BFCL test category
            temperature: Temperature parameter for the model
            backend: Backend used (vllm or sglang)
            cleanup_files: Whether to clean up files after evaluation
            
        Returns:
            Dictionary containing evaluation results and scores
        """
        try:
            # Set environment variables for the server
            os.environ["VLLM_ENDPOINT"] = endpoint
            os.environ["VLLM_PORT"] = str(port)
            
            # Import the generation function
            with self._silence_output():
                from bfcl._llm_response_generation import main as generation_main
            
            # Create directories if they don't exist
            os.makedirs(self.result_dir, exist_ok=True)
            os.makedirs(self.score_dir, exist_ok=True)
            
            # Create a standard path for model results
            model_dir_name = model_name.replace("/", "_")
            model_result_dir = os.path.join(self.result_dir, model_dir_name)
            model_score_dir = os.path.join(self.score_dir, model_dir_name)
            
            os.makedirs(model_result_dir, exist_ok=True)
            os.makedirs(model_score_dir, exist_ok=True)
            
            # Create arguments for generation
            gen_args = SimpleNamespace(
                model=[model_name],
                test_category=[test_category],
                temperature=temperature,
                include_input_log=False,
                exclude_state_log=False,
                num_gpus=1,
                num_threads=1,
                gpu_memory_utilization=0.9,
                backend=backend,
                skip_server_setup=True,  # Skip server setup since one is already running
                local_model_path=None,
                result_dir=str(self.result_dir),
                allow_overwrite=True,
                run_ids=False
            )
            
            # Run generation to create model responses (with silenced output if not verbose)
            with self._silence_output():
                generation_main(gen_args)
            
            # Verify result file exists
            result_file = os.path.join(model_result_dir, f"{self.VERSION_PREFIX}_{test_category}_result.json")
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"Result file not found: {result_file}")
            
            # Fix any decoding issues in the result file
            with self._silence_output():
                self._fix_decode_errors(result_file, test_category)
            
            # Load files and get handler with silenced output
            with self._silence_output():
                # Load the model results
                model_result = self.load_file(result_file, sort_by_id=True)
                
                # Find and load the prompt and possible answer files
                prompt_file = self.find_file_with_suffix(self.PROMPT_PATH, test_category)
                prompt = self.load_file(prompt_file, sort_by_id=True)
                
                # Find and load the possible answer file
                possible_answer_file = self.find_file_with_suffix(self.POSSIBLE_ANSWER_PATH, test_category)
                possible_answer = self.load_file(possible_answer_file, sort_by_id=True)
                
                # Get the appropriate handler for this model
                handler = self.get_handler(model_name)
            
            # Run the evaluation with silenced output
            with self._silence_output():
                # Call multi_turn_runner directly to get accuracy and total_count
                accuracy, total_count = self.multi_turn_runner(
                    handler,
                    model_result,
                    prompt,
                    possible_answer,
                    model_dir_name,  # Use directory name (with _ instead of /)
                    test_category,
                    self.score_dir
                )
            
            # Create minimal result dictionary
            result = {
                "overall_score": accuracy,
                "total_count": total_count,
                "correct_count": int(accuracy * total_count)
            }
            
            # Clean up if requested
            if cleanup_files:
                with self._silence_output():
                    self._clean_model_directories(model_dir_name)
            
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            self.log(f"BFCL evaluation failed: {error_trace}", "ERROR")
            return {
                "error": f"BFCL evaluation failed: {str(e)}",
                "overall_score": 0.0
            }
    
    def _fix_decode_errors(self, result_file: str, test_category: str) -> None:
        """
        Fix common decode errors in the result file.
        
        Args:
            result_file: Path to the result file
            test_category: Test category name
        """
        try:
            if not os.path.exists(result_file):
                self.log(f"Cannot fix decode errors: File not found - {result_file}", "WARNING")
                return
                
            with open(result_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    
                    # === For multi-turn tests ===
                    if self.is_multi_turn(test_category):
                        if "result" in data and isinstance(data["result"], list):
                            # This is the format for multi-turn data: list of turns, each with a list of steps
                            for i, turn_data in enumerate(data["result"]):
                                if isinstance(turn_data, list):
                                    for j, step_response in enumerate(turn_data):
                                        if isinstance(step_response, str) and "Failed to decode" in step_response:
                                            # Replace with a minimal valid function call for multi-turn
                                            function_match = re.search(r'\[(\w+)\(.*?\)\]', step_response)
                                            if function_match:
                                                # Use the actual function name if we can extract it
                                                function_name = function_match.group(1)
                                                data["result"][i][j] = f"[{function_name}(param=\"value\")]"
                                            else:
                                                data["result"][i][j] = "[function_name(param=\"value\")]"
                    
                    # === For single-turn tests ===
                    elif "result" in data and isinstance(data["result"], str):
                        if "Failed to decode" in data["result"]:
                            # Try to find function call patterns in the failed response
                            function_match = re.search(r'\[(\w+)\(.*?\)\]', data["result"])
                            if function_match:
                                # Use the actual function name if we can extract it
                                function_name = function_match.group(1)
                                data["result"] = f"[{function_name}(param=\"value\")]"
                            else:
                                # Default valid function call
                                data["result"] = "[function_name(param=\"value\")]"
                    
                    fixed_lines.append(json.dumps(data) + "\n")
                    
                except json.JSONDecodeError:
                    # Keep original line if we can't parse it
                    fixed_lines.append(line)
            
            # Write fixed content back to file
            with open(result_file, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
                
        except Exception as e:
            self.log(f"Error fixing decode errors: {str(e)}", "WARNING")
    
    def _clean_model_directories(self, model_name: str):
        """Clean up model-specific directories."""
        model_result_dir = os.path.join(self.result_dir, model_name)
        model_score_dir = os.path.join(self.score_dir, model_name)
        
        if os.path.exists(model_result_dir):
            shutil.rmtree(model_result_dir)
        
        if os.path.exists(model_score_dir):
            shutil.rmtree(model_score_dir)