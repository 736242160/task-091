"""逐字符驱动的字符串插值解析器（纯 Python 标准库，单文件）。

语法约定与设计理由
------------------
- 插值定界符: ``${`` 开、``}`` 闭（可用 parse() 的参数更换）。
  理由: 双字符开定界符在普通文本中出现概率低，与 shell / JS 模板字符串
  习惯一致；单字符闭定界符配合栈式嵌套即可正确配对。
- 插值内容按 **Python 表达式** 界定并用 ast.parse(mode="eval") 校验。
  理由: 复用标准库即可得到权威的位置化语法错误，无需自造表达式文法。
- 豁免（不计为定界符）:
  * 引号字符串 ``'...'`` / ``"..."``，支持反斜杠转义；
  * 行注释 ``#`` 到行尾。
- 歧义优先级: **字符串/注释豁免 > 插值定界符**。
  理由: 与主流词法器一致——字符串与注释是原子词法上下文，其内部字符
  不再参与外层结构匹配；否则 ``"${"`` 这类字面量将无法书写。
- 嵌套: 插值表达式中（非字符串/注释内）的 ``${`` 开启子插值，层级由栈维护。

输出
----
parse(text) -> (segments, errors)
- segments: 顶层交替列表，元素为 TextSegment / InterpSegment；
  InterpSegment.parts 同样是交替列表（嵌套结构），InterpSegment.expr 是
  重组后的表达式源码（嵌套插值替换为 __interp_N__ 占位符）。
- errors: ParseError 列表，含 kind / message / offset / line / col / fragment。
"""

from __future__ import annotations

import ast
import bisect
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 输出结构

@dataclass
class TextSegment:
    text: str
    start: int          # 起始偏移（含）
    end: int            # 结束偏移（不含）


@dataclass
class InterpSegment:
    start: int          # 开定界符 `$` 的偏移
    end: int            # 匹配闭定界符之后的位置（不含）
    parts: list         # 内部交替结构（TextSegment / InterpSegment）
    expr: str           # 重组后的表达式源码（嵌套插值 -> __interp_N__）
    depth: int          # 嵌套层级，顶层插值为 1


@dataclass
class ParseError:
    kind: str           # unclosed-interpolation / unclosed-string /
                        # invalid-expression / empty-expression
    message: str
    offset: int
    line: int
    col: int
    fragment: str | None = None


# ---------------------------------------------------------------- 内部状态

@dataclass
class _Frame:
    start: int                       # 开定界符起始偏移
    content_start: int               # 开定界符之后的偏移
    depth: int
    parts: list = field(default_factory=list)
    buf: list = field(default_factory=list)
    buf_start: int = 0
    quote: str | None = None         # 当前字符串引号（None 表示不在字符串内）
    quote_start: int = 0
    escaped: bool = False


def parse(text: str, open_delim: str = "${", close_delim: str = "}"):
    """逐字符推进的插值解析。返回 (segments, errors)。"""
    line_starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(idx + 1)

    def locate(offset: int):
        line = bisect.bisect_right(line_starts, offset) - 1
        return line + 1, offset - line_starts[line] + 1

    def make_error(kind, message, offset, fragment=None):
        line, col = locate(offset)
        return ParseError(kind, message, offset, line, col, fragment)

    segments: list = []
    errors: list[ParseError] = []
    stack: list[_Frame] = []

    text_buf: list[str] = []
    text_buf_start = 0

    def flush_text(pos):
        nonlocal text_buf, text_buf_start
        if text_buf:
            segments.append(TextSegment("".join(text_buf), text_buf_start, pos))
            text_buf = []

    def flush_expr(frame, pos):
        if frame.buf:
            frame.parts.append(
                TextSegment("".join(frame.buf), frame.buf_start, pos))
            frame.buf = []

    def build_expr(parts):
        chunks, n = [], 0
        for p in parts:
            if isinstance(p, TextSegment):
                chunks.append(p.text)
            else:
                chunks.append(f"__interp_{n}__")
                n += 1
        return "".join(chunks)

    def validate(frame, expr):
        stripped = expr.strip()
        if not stripped:
            errors.append(make_error(
                "empty-expression", "插值表达式为空", frame.content_start))
            return
        try:
            ast.parse(stripped, mode="eval")
        except SyntaxError as exc:
            # 将表达式内位置换算为源文本绝对位置（含嵌套占位符时为近似值）
            lines = stripped.splitlines(keepends=True)
            lineno = exc.lineno or 1
            off = sum(len(l) for l in lines[:lineno - 1]) + (exc.offset or 1) - 1
            errors.append(make_error(
                "invalid-expression",
                f"插值表达式语法非法: {exc.msg}",
                frame.content_start + off,
                fragment=stripped))

    n = len(text)
    i = 0
    while i < n:
        if not stack:
            # ---------------- TEXT 状态 ----------------
            if text.startswith(open_delim, i):
                flush_text(i)
                stack.append(_Frame(
                    start=i, content_start=i + len(open_delim),
                    depth=len(stack) + 1, buf_start=i + len(open_delim)))
                i += len(open_delim)
            else:
                if not text_buf:
                    text_buf_start = i
                text_buf.append(text[i])
                i += 1
        else:
            # ---------------- INTERP 状态 ----------------
            frame = stack[-1]
            ch = text[i]
            if frame.quote is not None:
                # 字符串内：一切定界符豁免，仅处理转义与收尾引号
                frame.buf.append(ch)
                if frame.escaped:
                    frame.escaped = False
                elif ch == "\\":
                    frame.escaped = True
                elif ch == frame.quote:
                    frame.quote = None
                i += 1
            elif ch in "'\"":
                frame.quote = ch
                frame.quote_start = i
                frame.buf.append(ch)
                i += 1
            elif ch == "#":
                # 行注释：到行尾为止全部豁免（注释文本保留在表达式中）
                j = i
                while j < n and text[j] != "\n":
                    j += 1
                frame.buf.append(text[i:j])
                i = j
            elif text.startswith(open_delim, i):
                # 嵌套插值
                flush_expr(frame, i)
                stack.append(_Frame(
                    start=i, content_start=i + len(open_delim),
                    depth=len(stack) + 1, buf_start=i + len(open_delim)))
                i += len(open_delim)
            elif text.startswith(close_delim, i):
                # 关闭当前插值
                flush_expr(frame, i)
                expr = build_expr(frame.parts)
                validate(frame, expr)
                node = InterpSegment(
                    start=frame.start, end=i + len(close_delim),
                    parts=frame.parts, expr=expr, depth=frame.depth)
                stack.pop()
                if stack:
                    stack[-1].parts.append(node)
                    stack[-1].buf_start = i + len(close_delim)
                else:
                    segments.append(node)
                i += len(close_delim)
            else:
                if not frame.buf:
                    frame.buf_start = i
                frame.buf.append(ch)
                i += 1

    flush_text(n)

    # 未闭合报告：每个仍在栈上的插值都报告其起始位置
    for frame in stack:
        errors.append(make_error(
            "unclosed-interpolation",
            f"插值定界符 {open_delim!r} 未闭合（起始于此处）",
            frame.start))
        if frame.quote is not None:
            errors.append(make_error(
                "unclosed-string",
                f"字符串引号 {frame.quote!r} 未闭合（起始于此处）",
                frame.quote_start))

    return segments, errors


# ---------------------------------------------------------------- 自测

def _test():
    # 1. 基本交替结构
    segs, errs = parse("hello ${name}!")
    assert not errs, errs
    assert [type(s).__name__ for s in segs] == [
        "TextSegment", "InterpSegment", "TextSegment"]
    assert segs[0].text == "hello " and segs[1].expr == "name"
    assert segs[2].text == "!"

    # 2. 嵌套层级
    segs, errs = parse("${ a + ${ b + ${ c } } }")
    assert not errs, errs
    outer = segs[0]
    assert outer.depth == 1
    mid = [p for p in outer.parts if isinstance(p, InterpSegment)]
    assert len(mid) == 1 and mid[0].depth == 2
    inner = [p for p in mid[0].parts if isinstance(p, InterpSegment)]
    assert len(inner) == 1 and inner[0].depth == 3 and inner[0].expr.strip() == "c"

    # 3. 字符串豁免：字符串内的 ${ 与 } 都不是定界符（歧义 -> 豁免优先）
    segs, errs = parse('${ "not ${interp} and } either" }')
    assert not errs, errs
    assert len(segs) == 1
    assert not any(isinstance(p, InterpSegment) for p in segs[0].parts)

    # 4. 转义引号
    segs, errs = parse("${ 'it\\'s } still string' }")
    assert not errs, errs and len(segs) == 1

    # 5. 注释豁免：注释里的定界符不算
    segs, errs = parse("${ x # } ${ ignored\n }")
    assert not errs, errs and len(segs) == 1

    # 6. 未闭合：报告起始位置
    segs, errs = parse("abc ${ def")
    assert [e.kind for e in errs] == ["unclosed-interpolation"]
    assert errs[0].offset == 4 and (errs[0].line, errs[0].col) == (1, 5)

    # 7. 非法表达式：报告片段与位置
    segs, errs = parse("${ 1 + }")
    assert [e.kind for e in errs] == ["invalid-expression"]
    assert errs[0].fragment == "1 +"
    assert errs[0].offset >= 2

    # 8. 空表达式
    segs, errs = parse("${}")
    assert [e.kind for e in errs] == ["empty-expression"]

    # 9. 多层未闭合：每层都报告起始位置
    segs, errs = parse("${ a ${ b")
    kinds = [e.kind for e in errs]
    assert kinds == ["unclosed-interpolation", "unclosed-interpolation"]
    assert errs[0].offset == 0 and errs[1].offset == 5

    # 10. 未闭合字符串
    segs, errs = parse("${ 'abc")
    kinds = [e.kind for e in errs]
    assert "unclosed-interpolation" in kinds and "unclosed-string" in kinds

    # 11. 多行文本的行列号
    segs, errs = parse("line1\nline2 ${ x")
    assert (errs[0].line, errs[0].col) == (2, 7)

    # 12. 无插值纯文本
    segs, errs = parse("plain } text $ {")
    assert not errs and len(segs) == 1 and segs[0].text == "plain } text $ {"

    print("全部自测通过 ✔")


if __name__ == "__main__":
    _test()

    demo = '你好 ${ user.name.upper() + "（${不是插值}）" }，余额 ${ f(${base} * 2) # 注释里的 } 不算\n } 元'
    segs, errs = parse(demo)
    print("\n== 演示输入 ==")
    print(demo)
    print("\n== 解析结构 ==")

    def show(parts, indent=0):
        for p in parts:
            pad = "  " * indent
            if isinstance(p, TextSegment):
                print(f"{pad}TEXT [{p.start}:{p.end}] {p.text!r}")
            else:
                print(f"{pad}INTERP depth={p.depth} [{p.start}:{p.end}] expr={p.expr!r}")
                show(p.parts, indent + 1)

    show(segs)
    print("\n== 错误清单 ==")
    for e in errs:
        print(f"{e.kind} @行{e.line}列{e.col}: {e.message} fragment={e.fragment!r}")
    if not errs:
        print("（无）")

    bad = "未闭合: ${ 1 +"
    segs, errs = parse(bad)
    print(f"\n== 错误演示 {bad!r} ==")
    for e in errs:
        print(f"{e.kind} @行{e.line}列{e.col} 偏移{e.offset}: {e.message} fragment={e.fragment!r}")
