"""Production answer for the unicode2 question."""

a_answer = (
    "UTF-8 is usually preferred because it is the dominant web encoding, is more space-efficient "
    "than UTF-16 or UTF-32 for ASCII-heavy text, and avoids endianness and BOM complications that "
    "arise with UTF-16 and UTF-32."
)

b_example = "b'\\xc3\\xa9'"
b_answer = (
    "For example, the UTF-8 byte string b'\\xc3\\xa9' (which should decode to 'é') breaks this "
    "function because it decodes each byte separately even though UTF-8 characters may span multiple "
    "bytes, so individual bytes like 0xC3 and 0xA9 are not valid standalone UTF-8 characters."
)

c_example = "b'\\xff\\xff'"
c_answer = (
    "For example, b'\\xff\\xff' does not decode to any Unicode character(s) in UTF-8 because 0xFF "
    "is never a valid UTF-8 byte."
)


if __name__ == "__main__":
    print("(a)", a_answer)
    print("(b)", b_answer)
    print("(c)", c_answer)
