import argparse
import onnx
import numpy as np
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir
from tvm.contrib import ndk

parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx_path", type=str, help="Path of the onnx model."
)
parser.add_argument(
    "--bsr", type=int, help="Size of Sparse Block(row)."
)
parser.add_argument(
    "--bsc", type=int, help="Size of Sparse Block(col)."
)
parser.add_argument(
    "--sparsity", type=int, help="The Sparsity of Network."
)

args = parser.parse_args()

def main():
    # Get Model
    print("Get Model...")
    onnx_model = onnx.load(args.onnx_path)
    shape_dict = {}
    for input in onnx_model.graph.input:
        shape_dict[input.name] = [
            dim.dim_value for dim in input.type.tensor_type.shape.dim
        ]
    mod, params = relay.frontend.from_onnx(onnx_model)

    bs_r = args.bsr
    bs_c = args.bsc
    sparsity = args.sparsity

    # Conver to Sparse Model
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod, params = ddo.bsr_conv2d.convert(
        mod, params, (bs_r, bs_c), sparsity_threshold=sparsity, layout='NHWC'
    )
    mod = tvm.IRModule.from_expr(mod)

    # Set tune config
    target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
    device_key = "pixel2"
    rpc_host = "127.0.0.1"
    rpc_port = 9190

    log_file = f"{str(args.onnx_path).split('.')[-2]}.json"

    # Extract tasks
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    def tune_and_evaluate():
        print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            builder=auto_scheduler.LocalBuilder(build_func="ndk"),
            runner=auto_scheduler.RPCRunner(
                device_key,
                host=rpc_host,
                port=rpc_port,
                timeout=30,
                repeat=1,
                min_repeat_ms=200,
                enable_cpu_cache_flush=True,
            ),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)

        # Compile with the history best
        print("Compile...")
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)

        # Export library
        tmp = tempdir()
        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
        
        # Upload module to device
        print("Upload...")
        remote = auto_scheduler.utils.request_remote(device_key, rpc_host, rpc_port, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # Create graph executor
        dev = remote.cpu()
        module = graph_executor.GraphModule(rlib["default"](dev))
        for key, value in shape_dict.items():
            data_tvm = tvm.nd.array((np.random.uniform(size=value)).astype("float32"))
            module.set_input(key, data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
        )
    
    tune_and_evaluate()

if __name__ == "__main__":
    main()
