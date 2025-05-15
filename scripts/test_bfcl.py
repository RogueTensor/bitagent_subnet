
print("Testing BFCL installation... This should take less than 30 seconds...")

try:
    # Silence import noise if not verbose

        from bfcl.constants.category_mapping import VERSION_PREFIX
        from bfcl.constants.eval_config import (
            PROMPT_PATH, PROJECT_ROOT, POSSIBLE_ANSWER_PATH,
            RESULT_PATH, SCORE_PATH
        )
        from bfcl.utils import (
            load_file, find_file_with_suffix, is_multi_turn
        )
        

        # Import just the multi_turn_runner and get_handler functions
        from bfcl.eval_checker.eval_runner import (
            multi_turn_runner, get_handler
        )
        
        print("BFCL modules imported successfully!")
    
except ImportError as e:
    print(f"[ERROR] Could not import BFCL modules: {e}")
    print(f"[ERROR] Make sure BFCL is installed or the path is correct")
    raise ImportError(f"Required BFCL modules not found: {str(e)}")