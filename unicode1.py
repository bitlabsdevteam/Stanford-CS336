"""Production answer for the unicode1 question."""

a_answer = "chr(0) returns the Unicode NULL character, whose code point is U+0000."

b_answer = "Its string representation shows an escaped form like '\\x00', while its printed representation appears blank because NULL is a non-printing character."

c_answer = "When this character occurs in Python text, it is stored as part of the string but produces no visible output when printed, so the surrounding text looks unchanged."


if __name__ == "__main__":
    print("(a)", a_answer)
    print("(b)", b_answer)
    print("(c)", c_answer)
