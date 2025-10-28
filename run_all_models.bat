@ECHO OFF
SETLOCAL

:: --- 配置 ---
:: 1. 设置你的 Conda 环境名称
SET "ENV_NAME=pytorch_env"

:: 2. 设置训练脚本的路径 (假设此 .bat 文件与 train.py 在同一目录)
SET "SCRIPT_PATH=train.py"

:: 3. 设置训练的 Epochs (设为较小的值, 比如 1, 仅用于测试是否能跑通)
SET "EPOCHS=150"
:: --- 配置结束 ---

ECHO 启动模型性能测试循环...
ECHO Conda 环境: %ENV_NAME%
ECHO 训练脚本: %SCRIPT_PATH%

:: 循环遍历所有模型。
:: 关键在于 "(... || ...)" 结构：
:: 如果括号内的 conda run 命令失败 (返回非零错误代码),
:: 脚本将执行 "||" 右侧的 ECHO 语句, 然后继续执行 FOR 循环的下一次迭代。
FOR %%M IN (bilstm mlp transformer hier) DO (
    ECHO.
    ECHO ===================================================
    ECHO 正在启动模型: %%M
    ECHO ===================================================

    (
        ECHO 正在运行: conda run -n %ENV_NAME% --no-capture-output python %SCRIPT_PATH% --model %%M --epochs %EPOCHS%
        conda run -n %ENV_NAME% --no-capture-output python %SCRIPT_PATH% --model %%M --epochs %EPOCHS%
    ) || (
        ECHO.
        ECHO [!!! 错误 !!!] - 模型 '%%M' 运行失败。跳过并继续...
        ECHO ===================================================
    )
)

ECHO.
ECHO ===================================================
ECHO 所有模型测试完成。
ECHO ===================================================
ENDLOCAL
PAUSE