"""Verify archive loading leaves every stored policy tensor unchanged."""
from pathlib import Path
import sys
import io
import json
import hashlib
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from stable_baselines3 import PPO,SAC,TD3
    from multi_algorithm_benchmark.archive_compat import load_archived_model
    torch.set_num_threads(1)
    archive=ROOT/"01_遴选原始参考"/"05_中期前任务收尾新增材料"/"原样工程"/"大一立项AI在卫星姿态调整中的应用"
    paths=[("single",PPO,archive/"单轴强化学习"/"ppo_satellite_final.zip"),
           ("three",PPO,archive/"三轴强化学习"/"ppo_satellite3d.zip"),
           ("three",SAC,archive/"三轴强化学习"/"sac_satellite3d.zip"),
           ("three",TD3,archive/"三轴强化学习"/"td3_satellite3d.zip")]
    output=[]
    for axis,algorithm,path in paths:
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with ZipFile(path) as z:
            metadata=json.loads(z.read("data"))
            stored=torch.load(io.BytesIO(z.read("policy.pth")),map_location="cpu",weights_only=True)
        model=load_archived_model(algorithm,path)
        restored=model.policy.state_dict()
        assert set(stored)==set(restored),"Policy tensor keys changed"
        tensor_rows=[]
        for key,value in stored.items():
            actual=restored[key].detach().cpu()
            assert value.shape==actual.shape and value.dtype==actual.dtype,key
            expected_bytes=value.detach().cpu().contiguous().numpy().tobytes()
            actual_bytes=actual.contiguous().numpy().tobytes()
            assert expected_bytes==actual_bytes,f"Policy tensor bytes changed: {key}"
            tensor_rows.append({"name":key,"shape":list(value.shape),"dtype":str(value.dtype),
                                "sha256":hashlib.sha256(actual_bytes).hexdigest(),"byte_identical":True})
        spaces={}
        for key in ("action_space","observation_space"):
            raw=metadata[key];space=getattr(model,key)
            shape=tuple(raw["_shape"]);dtype=np.dtype(raw["dtype"])
            assert shape==space.shape and dtype==space.dtype,key
            for bound in ("low","high"):
                expected=np.asarray([float(number) for number in raw[bound].strip("[]").split()],dtype=dtype).reshape(shape)
                actual=getattr(space,bound)
                assert expected.dtype==actual.dtype and expected.tobytes()==actual.tobytes(),key+" "+bound
            spaces[key]={"shape":list(space.shape),"dtype":str(space.dtype),
                         "low":space.low.tolist(),"high":space.high.tolist(),"metadata_identical":True,
                         "saved_unused_rng_omitted":True}
        zero=np.zeros((1,*model.observation_space.shape),dtype=model.observation_space.dtype)
        action,_=model.predict(zero,deterministic=True)
        assert np.isfinite(action).all(),"Inference action not finite"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before,"Source ZIP changed"
        record={"axis":axis,"algorithm":algorithm.__name__,"zip_sha256":before,
                "source_zip_unchanged":True,"tensor_count":len(tensor_rows),"tensors":tensor_rows,
                "spaces":spaces,"zero_observation_physical_action":action.tolist(),
                "actual_training_steps":int(model.num_timesteps),"finite_inference":True}
        output.append(record)
        print(axis,algorithm.__name__,"all",len(tensor_rows),"policy tensors byte identical; Box metadata identical",flush=True)
    report={"status":"passed","test_results_read":False,"policy_tensor_byte_identical":True,
            "space_bounds_dtype_shape_identical":True,"zip_files_modified":False,
            "loading_adapter_only":"NumPy 2 metadata class module remapping and Box reconstruction; unused serialized space RNG omitted",
            "adapter_source_sha256":hashlib.sha256((ROOT/"multi_algorithm_benchmark"/"archive_compat.py").read_bytes()).hexdigest(),
            "models":output}
    (Path(__file__).resolve().parent/"历史模型兼容加载复核.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":main()
