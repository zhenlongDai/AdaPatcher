import re

def remove_comments(code):
    """
    移除代码中的注释，包括单行注释和多行注释
    :param code: 包含注释的代码字符串
    :return: 移除注释后的代码字符串
    """
    # 定义用于匹配单行和多行注释的正则表达式
    single_line_comment_pattern = r'//.*?$|#.*?$'
    multi_line_comment_pattern = r'/\*.*?\*/|\'\'\'.*?\'\'\'|""".*?"""'
    
    # 组合正则表达式
    pattern = re.compile(
        single_line_comment_pattern + '|' + multi_line_comment_pattern,
        re.DOTALL | re.MULTILINE
    )

    # 使用正则表达式移除注释
    cleaned_code = re.sub(pattern, '', code)
    
    return cleaned_code

# 示例代码字符串
code_with_comments = "import sys\n\n#\u6700\u5927\u516c\u7d04\u6570\ndef gcd(x, y):\n    s = x / y\n    r = x % y\n    if r == 0:\n        return y\n    else:\n        return gcd(y, r)\n\n#\u6700\u5c0f\u516c\u500d\u6570\ndef lcm(x, y):\n    return x*y/gcd(x, y)\n\ndef main():\n    #print (\"a\")\n    for line in iter(sys.stdin.readline, \"\"):\n        print (line)\n        #tmp = sys.stdin.readline().split(\" \")\n        #print (\"b\")\n        tmp = line.split(\" \")\n    \n        a = int(tmp[0])\n        b = int(tmp[1])\n        #print (\"a=\"+str(a))\n        #print (\"b=\"+str(b))\n        #b = sys.stdin.readline()\n        #print (\"d\")\n        if a > b:\n            c = a\n            d = b\n        else:\n            c = b\n            d = a\n\n        print (str(gcd(c, d)) + \" \" + str(int(lcm(c,d))))\n\n        #tmp = sys.stdin.readline()\n        #if len(tmp) == 1:\n        #    break\n        #else:\n        #    tmp = tmp.split(\" \")\n\n    #print (\"exit\")\n     \nif __name__ == \"__main__\":\n    main()"

# 移除注释
cleaned_code = remove_comments(code_with_comments)

# 打印移除注释后的代码
print("ok")
print(cleaned_code)

