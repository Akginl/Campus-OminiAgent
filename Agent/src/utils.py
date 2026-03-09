import inspect


def function_to_json(func) -> dict:
    # 定义Python类型到JSON数据类型的映射
    type_map = {
        str: "string",
        int: "integer",
        float: "float",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    # 获取函数的签名信息
    try:
        # inspect获取可调用对象的参数信息，返回一个Signature对象
        # 包含参数名称，类型注解，默认值等信息
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果获取签名失败，抛出具体的错误信息
        raise ValueError(
            f"无法获取函数{func.__name__}的签名{str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        # 尝试获取参数的类型，如果无法找到对应的类型则默认设置为string
        try:
            # param.annotation是Parameter对象的属性
            # 表示参数的注解，如 int str float等
            # 调用字典的get方法，第一个参数是注解对象
            # 第二个参数是默认值，如果字典中存在该键，则返回对应的值，不存在则返回string
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,  # 函数参数的描述类型
                "required": required,  # 函数中没有默认值的参数列表
            }
        }
    }


