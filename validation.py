import re

def validate_input(user_input: str):
    """验证和清洗用户输入，返回 (is_valid, cleaned_or_error_msg)"""
    if len(user_input) > 1000:
        return False, "输入过长，请缩短您的问题"

    injection_patterns = [
        r"忽略.*指令", r"忘记.*之前", r"新.*指令", r"覆盖.*系统",
        r"扮演.*角色", r"开发者.*模式", r"系统.*消息", r"提示词",
        r"系统提示", r"重置.*规则", r"忽略.*规则", r"修改.*规则",
        r"输出.*指令", r"打印.*指令", r"显示.*指令", r"告诉.*指令",
        r"泄露.*指令", r"透露.*指令"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, user_input, re.I):
            return False, "检测到可疑输入，请重新表述您的问题"

    cleaned = re.sub(r"[<>{}[\]\\]", "", user_input)
    return True, cleaned