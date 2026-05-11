import json
import traceback
from .utils import check_correctness as apps_check_correctness

def compute_score(completion, test_cases, continuous=False):
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases, generation=solution, timeout=5, debug=False
            )
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass
        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})
        if continuous:
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(
                    in_outs=test_case, generation=solution, timeout=10, debug=False
                )
                try:
                    metadata = dict(enumerate(metadata))[0]
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)
                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
