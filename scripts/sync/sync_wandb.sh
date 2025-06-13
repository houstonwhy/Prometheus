#!/bin/bash

# 定义输入目录和OSS目标目录
C2O_INPUT_DIR="/input/yyb/PlatonicGen/outputs/logs"
C2O_OSS_DIR="oss://antsys-vilab/yyb//PlatonicGen/outputs/logs"

O2L_INPUT_DIR="oss://antsys-vilab/yyb//PlatonicGen/outputs/logs"
O2L_OSS_DIR="/data1/yyb/PlatonicGen/outputs/logs"

# 帮助信息
usage() {
    echo "Usage: $0 -m <mode> -j <jobid>"
    echo "  -m <mode>    : Specify the mode (c2o for local to OSS, o2l for OSS to local)."
    echo "  -j <jobid>   : Specify the job ID."
    exit 1
}

# 解析命令行参数
while getopts "m:j:" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        j) JOBID="$OPTARG" ;;
        *) usage ;;
    esac
done

# 检查是否提供了mode和jobid
if [ -z "$MODE" ] || [ -z "$JOBID" ]; then
    echo "Error: Mode and Job ID are required."
    usage
fi

# 根据mode设置输入目录和OSS目标目录
case $MODE in
    c2o)
        INPUT_DIR="$C2O_INPUT_DIR"
        OSS_DIR="$C2O_OSS_DIR"
        ;;
    o2l)
        INPUT_DIR="$O2L_INPUT_DIR"
        OSS_DIR="$O2L_OSS_DIR"
        ;;
    *)
        echo "Error: Invalid mode. Use 'c2o' for local to OSS or 'o2l' for OSS to local."
        usage
        ;;
esac

# 构建输入目录路径
JOB_DIR="$INPUT_DIR/$JOBID"

# # 检查目录是否存在
# if [ ! -d "$JOB_DIR" ]; then
#     echo "Error: Directory $JOB_DIR does not exist."
#     exit 1
# fi

# 构建OSS目标路径
OSS_TARGET_DIR="$OSS_DIR/$JOBID"

# 同步目录到OSS
echo "Syncing $JOB_DIR to $OSS_TARGET_DIR"
ossutil64 cp -r -f "$JOB_DIR/" "$OSS_TARGET_DIR/"

echo "Sync completed."