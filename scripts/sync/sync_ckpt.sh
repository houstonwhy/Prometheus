#!/bin/bash

# 定义输入目录和OSS目标目录
C2O_INPUT_DIR="/input/yyb/PlatonicGen/outputs/ckpts"
C2O_OSS_DIR="oss://antsys-vilab/yyb/PlatonicGen/outputs/ckpts"

O2L_INPUT_DIR="oss://antsys-vilab/yyb/PlatonicGen/outputs/ckpts"
O2L_OSS_DIR="/data1/yyb/PlatonicGen/outputs/ckpts"

# 帮助信息
usage() {
    echo "Usage: $0 -m <mode> -j <jobid> [-s <stepid>]"
    echo "  -m <mode>    : Specify the mode (c2o for local to OSS, o2l for OSS to local)."
    echo "  -j <jobid>   : Specify the job ID."
    echo "  -s <stepid>  : Specify the step ID. Required for o2l mode."
    exit 1
}

# 解析命令行参数
while getopts "m:j:s:" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        j) JOBID="$OPTARG" ;;
        s) STEPID="$OPTARG" ;;
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
        # 检查是否提供了stepid
        if [ -z "$STEPID" ]; then
            echo "Error: Step ID is required for o2l mode."
            usage
        fi
        ;;
    *)
        echo "Error: Invalid mode. Use 'c2o' for local to OSS or 'o2l' for OSS to local."
        usage
        ;;
esac

# 构建输入目录路径
JOB_DIR="$INPUT_DIR/$JOBID"

# 如果没有指定stepid，找到step最大的文件
if [ -z "$STEPID" ]; then
    # 找到所有以.ckpt结尾的文件，并按step排序
    LATEST_FILE=$(ls -1v "$JOB_DIR"/*.ckpt 2>/dev/null | grep -oP 'step=\K[0-9]+' | sort -n | tail -1)
    if [ -z "$LATEST_FILE" ]; then
        echo "Error: No checkpoint files found in $JOB_DIR."
        exit 1
    fi
    STEPID="$LATEST_FILE"
fi

# 构建文件路径
if [ "$MODE" == "c2o" ]; then
    FILE_PATH=$(find "$JOB_DIR" -name "*step=$STEPID*.ckpt" 2>/dev/null)
else
    FILE_PATH="$JOB_DIR/$STEPID.ckpt"
fi

# 检查文件是否存在
if [ -z "$FILE_PATH" ]; then
    echo "Error: No checkpoint file found for step $STEPID in $JOB_DIR."
    exit 1
fi

# 构建OSS目标路径
OSS_TARGET_DIR="$OSS_DIR/$JOBID"

# 同步文件到OSS
echo "Syncing $FILE_PATH to $OSS_TARGET_DIR"
ossutil64 cp "$FILE_PATH" "$OSS_TARGET_DIR/"

echo "Sync completed."