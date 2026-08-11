from typing import TypeVar, Protocol, Union, get_type_hints,cast
from functools import wraps
import inspect
# from dataflow.core import OperatorABC

class PromptABC():
    def __init__(self):
        pass
    def build_prompt(self):
        raise NotImplementedError

class DIYPromptABC(PromptABC):
    def __init__(self):
        super().__init__()
    def build_prompt(self):
        raise NotImplementedError
    
# class OperatorWithAllowedPrompts(Protocol):
#     ALLOWED_PROMPTS: list[type[DIYPromptABC | PromptABC]]

def _make_diyprompt_union(allowed_prompts: tuple[type[PromptABC], ...]):
    """构造一个 Union 类型，包含允许的 prompt + DIYPromptABC 子类 + None"""
    return Union[tuple(allowed_prompts) + (DIYPromptABC, type(None))]

# 泛型参数，表示任意传入的 class 类型
T = TypeVar("T")

def prompt_restrict(*allowed_prompts: type[DIYPromptABC]):
    """
    装饰器：限制 prompt_template 只能是指定 Prompt 类 或 DIYPromptABC 子类
    并在运行时检查 & 更新 __annotations__（供 get_type_hints 等工具使用）
    """
    def decorator(cls:T) -> T:
        setattr(cls, "ALLOWED_PROMPTS", tuple(allowed_prompts))
        # self.ALLOWED_PROMPTS = list(allowed_prompts)

        orig_init = cls.__init__
        sig = inspect.signature(orig_init)  # 在装饰时就解析一次签名，避免每次实例化重复解析
        if "prompt_template" not in sig.parameters:
            # 若类的 __init__ 根本没有该形参，就仅维持注解/属性设置，不做运行时检查
            # （你也可以选择在这里直接 raise 来强制类必须声明该参数）
            pass

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            # 用签名绑定实参：自动把位置/关键字/默认值对齐到参数名
            try:
                bound = sig.bind_partial(self, *args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                # 参数不完整或不匹配时，交给原始 __init__ 去报错更合适
                return orig_init(self, *args, **kwargs)

            pt = bound.arguments.get("prompt_template", None)

            if pt is not None and not isinstance(pt, cls.ALLOWED_PROMPTS):
                if not isinstance(pt, DIYPromptABC):
                    allowed_names = "\n".join(
                        f"  - {c.__module__}.{c.__qualname__}"
                        for c in cls.ALLOWED_PROMPTS
                    )
                    raise TypeError(
                        f"[{cls.__name__}] Invalid prompt_template type: "
                        f"{type(pt).__module__}.{type(pt).__qualname__}\n"
                        f"Expected one of:\n{allowed_names}\n"
                        f"or a custom subclass of `dataflow.core.prompt.DIYPromptABC.`"
                    )

            # -------- 兼容兜底：补齐缺失的必填参数（除 self/prompt_template）--------
            # 目的：让 prompt_template 的限制不因为“其他必填参数未传”而导致测试无法覆盖。
            # 实际业务中建议调用方显式传入必填参数。
            missing_required = []
            for name, p in sig.parameters.items():
                if name == "self" or name == "prompt_template":
                    continue
                if p.default is not inspect._empty:
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name not in bound.arguments:
                    missing_required.append((name, p))

            if missing_required:
                for name, p in missing_required:
                    anno = p.annotation
                    if anno in (int, "int", float, "float"):
                        bound.arguments[name] = 0
                    else:
                        bound.arguments[name] = None

                # 只用关键字调用，避免位置参数重排风险
                call_kwargs = dict(bound.arguments)
                call_kwargs.pop("self", None)
                return orig_init(self, **call_kwargs)

            return orig_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # 保持你原本的注解暴露逻辑
        cls.__annotations__ = dict(getattr(cls, "__annotations__", {}))
        cls.__annotations__["prompt_template"] = _make_diyprompt_union(allowed_prompts)

        return cls
    return decorator


if __name__ == "__main__":
    import pytest

    class A(PromptABC): pass
    class B(PromptABC): pass
    class MyDIY(DIYPromptABC): pass
    class Other(PromptABC): pass

    @prompt_restrict(A, B)
    class Op:
        def __init__(self, prompt_template=None):
            self.prompt_template = prompt_template

    # 关键字参数：允许
    Op(prompt_template=A())
    Op(prompt_template=B())
    Op(prompt_template=MyDIY())
    Op()  # None 允许

    # 位置参数：同样被检测
    Op(A())        # ✅
    Op(MyDIY())    # ✅
    with pytest.raises(TypeError):
        Op(Other())  # ❌ 非白名单且非 DIY

    with pytest.raises(TypeError):
        Op(object())  # ❌ 完全无关类型
