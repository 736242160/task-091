#!/usr/bin/env python3
"""逐字符驱动的嵌套字符串插值解析器（纯标准库，单文件）。

语法约定
--------
- 插值定界符:  `${` 开, `}` 闭。
- 转义:        `\\${` 表示字面量 `${`（仅在普通文本/表达式文本中）。
- 表达式:      合法 Python 表达式（用 ast.parse(mode="eval") 校验）。
- 嵌套:        表达式内可再次出现 `${ ... }`。
- 豁免区:      表达式内的 '...' / "..." 字符串与 # 至行尾的注释，
               其中的 ${ 和 } 不参与解析。

歧义优先级（同一位置多条规则同时适用时）
--------------------------------------
转义 > 字符串/注释豁免 > 插值开符 > 插值闭符。
理由: 转义是最局部的显式意图；字符串/注释一旦进入，其内容在词法上
不可再被解释；闭符优先级最低，仅作为兜底结束当前插值层。

输出
----
parse(text) -> (segments, errors)
segments: TextSeg / InterpSeg 交替列表；InterpSeg.parts 递归同构。
errors:   ParseError 列表（未闭合插值 / 非法表达式 / 未闭合字符串）。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union

OPEN = "${"
CLOSE = "}"
ESC = "\\"


# ---------------------------------------------------------------- 输出结构

@dataclass
class TextSeg:
    text: str
    start: int
    kind: str = field(default="text", init=False)


@dataclass
class InterpSeg:
    start: int                 # `${` 的位置
    parts: List["Segment"]     # 表达式内容：文本片与嵌套 InterpSeg 交替
    closed: bool
    end: Optional[int] = None  # 匹配 `}` 之后的位置；未闭合为 None
    kind: str = field(default="interp", init=False)


Segment = Union[TextSeg, InterpSeg]


@dataclass
class ParseError:
    kind: str                  # unclosed_interpolation / bad_expression / unterminated_string
    pos: int                   # 绝对偏移
    line: int
    col: int
    message: str
    fragment: str = ""         # 相关源码片段（如非法表达式）


# ---------------------------------------------------------------- 解析器

class Parser:
    """显式状态机：TEXT 与 INTERP 两态，逐字符推进。

    INTERP 态内部再叠加 STRING / COMMENT 子状态（豁免区）。
    嵌套通过递归调用 _parse_segments 实现，每层独立计数。
    """

    def __init__(self, src: str):
        self.s = src
        self.n = len(src)
        self.i = 0
        self.errors: List[ParseError] = []

    # -- 工具 ----------------------------------------------------------

    def _line_col(self, pos: int) -> Tuple[int, int]:
        line = self.s.count("\n", 0, pos) + 1
        col = pos - (self.s.rfind("\n", 0, pos) + 1) + 1
        return line, col

    def _err(self, kind: str, pos: int, message: str, fragment: str = ""):
        line, col = self._line_col(pos)
        self.errors.append(ParseError(kind, pos, line, col, message, fragment))

    # -- 主入口 --------------------------------------------------------

    def parse(self) -> List[Segment]:
        segs, _ = self._parse_segments(in_interp=False)
        return segs

    # -- 状态机核心 ----------------------------------------------------

    def _parse_segments(self, in_interp: bool) -> Tuple[List[Segment], bool]:
        """逐字符推进 TEXT 态；返回 (段列表, 是否遇到匹配的闭符)。"""
        segs: List[Segment] = []
        buf: List[str] = []
        buf_start = self.i

        def flush():
            nonlocal buf_start
            if buf:
                segs.append(TextSeg("".join(buf), buf_start))
                buf.clear()

        while self.i < self.n:
            c = self.s[self.i]

            # 优先级 1: 转义 \${ —— 最局部的显式意图
            if c == ESC and self.s.startswith(OPEN, self.i + 1):
                buf.append(OPEN)
                self.i += 1 + len(OPEN)
                continue

            # 优先级 2: 豁免区（仅插值表达式内有字符串/注释概念）
            if in_interp and c in "'\"":
                buf.append(c)
                self.i += 1
                self._consume_string(c, buf)
                continue
            if in_interp and c == "#":
                while self.i < self.n and self.s[self.i] != "\n":
                    buf.append(self.s[self.i])
                    self.i += 1
                continue

            # 优先级 3: 插值开符（嵌套）
            if self.s.startswith(OPEN, self.i):
                flush()
                segs.append(self._parse_interp())
                buf_start = self.i
                continue

            # 优先级 4: 插值闭符 —— 结束当前层
            if in_interp and c == CLOSE:
                flush()
                return segs, True

            buf.append(c)
            self.i += 1

        flush()
        return segs, False

    def _consume_string(self, quote: str, buf: List[str]):
        """STRING 子状态：吞掉整个字符串字面量（豁免区）。"""
        start = self.i - 1
        while self.i < self.n:
            c = self.s[self.i]
            if c == ESC and self.i + 1 < self.n:
                buf.append(self.s[self.i:self.i + 2])
                self.i += 2
                continue
            buf.append(c)
            self.i += 1
            if c == quote:
                return
        self._err("unterminated_string", start,
                  "表达式内字符串字面量未闭合",
                  fragment=self.s[start:start + 40])

    def _parse_interp(self) -> InterpSeg:
        """消费 `${`，递归解析内部（INTERP 态），再消费 `}`。"""
        start = self.i
        self.i += len(OPEN)
        err_mark = len(self.errors)
        parts, closed = self._parse_segments(in_interp=True)
        end = None
        if closed:
            self.i += 1  # 消费 '}'
            end = self.i
        else:
            self._err("unclosed_interpolation", start,
                      "插值定界符 '${' 未闭合",
                      fragment=self.s[start:start + 40])
        seg = InterpSeg(start=start, parts=parts, closed=closed, end=end)
        # 结构完好的插值才做表达式语法校验，避免级联重复报错
        if closed and len(self.errors) == err_mark:
            self._check_expr(seg)
        return seg

    # -- 表达式语法校验 --------------------------------------------------

    def _expr_source(self, seg: InterpSeg) -> str:
        """重建表达式源码：嵌套插值替换为等长占位标识符（保持列号不漂移）。"""
        out = []
        for idx, p in enumerate(seg.parts):
            if isinstance(p, TextSeg):
                out.append(p.text)
            else:
                span = (p.end if p.end is not None else self.n) - p.start
                name = "_i%d" % idx
                out.append(name + "_" * max(0, span - len(name)))
        return "".join(out)

    def _check_expr(self, seg: InterpSeg):
        expr = self._expr_source(seg)
        expr_start = seg.start + len(OPEN)
        stripped = expr.strip()
        lead = len(expr) - len(expr.lstrip())  # strip 掉的前缀长度，用于位置补偿
        try:
            ast.parse(stripped, mode="eval")
        except SyntaxError as e:
            # 把表达式内的 (lineno, offset) 映射回源文本绝对位置
            pos = expr_start + lead
            if e.lineno is not None:
                lines = stripped.splitlines(keepends=True)
                pos += sum(len(l) for l in lines[:e.lineno - 1])
                pos += (e.offset or 1) - 1
            self._err("bad_expression", min(pos, self.n),
                      "插值表达式语法非法: %s" % (e.msg or "invalid syntax"),
                      fragment=stripped[:60])


def parse(text: str) -> Tuple[List[Segment], List[ParseError]]:
    p = Parser(text)
    segs = p.parse()
    return segs, p.errors


# ---------------------------------------------------------------- 自测

def _show(segs, indent=0):
    pad = "  " * indent
    for s in segs:
        if isinstance(s, TextSeg):
            print("%sTEXT   @%d  %r" % (pad, s.start, s.text))
        else:
            print("%sINTERP @%d  closed=%s" % (pad, s.start, s.closed))
            _show(s.parts, indent + 1)


def _run(title, text, expect_errors=0):
    print("=" * 60)
    print("%s\n输入: %r" % (title, text))
    segs, errors = parse(text)
    _show(segs)
    for e in errors:
        print("ERROR  %s @%d (行%d列%d): %s 片段=%r"
              % (e.kind, e.pos, e.line, e.col, e.message, e.fragment))
    assert len(errors) == expect_errors, (title, errors)
    print("OK (错误数=%d)" % len(errors))


if __name__ == "__main__":
    _run("1 纯文本", "hello, world")
    _run("2 基本插值", "hello ${name}!")
    _run("3 嵌套插值", "和: ${ 1 + ${ 2 + ${ 3 } } }")
    _run("4 字符串豁免", r'${ "字面 ${ 不是插值 }" + "}" }')
    _run("5 注释豁免", "${ x  # 注释里的 ${ 和 } 都不算\n}")
    _run("6 转义", r"字面 \${ 不插值 } 与 ${real}")
    _run("7 未闭合插值", "abc ${ 1 + 2", expect_errors=1)
    _run("8 非法表达式", "值=${ 1 + }", expect_errors=1)
    _run("9 嵌套未闭合", "${ 外层 ${ 内层 } ", expect_errors=1)  # 未闭合（结构错误时跳过表达式校验）
    _run("10 空表达式", "x${}y", expect_errors=1)
    _run("11 未闭合字符串", "${ 'abc ", expect_errors=2)  # 字符串未闭合+插值未闭合
    _run("12 混合", "a${ f('${x}') + ${ y } }b")
    print("=" * 60)
    print("全部自测通过")
