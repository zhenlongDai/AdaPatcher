import sys

for i in range(int(input())):   # (0): i=0
    x1, y1, x2, y2, x3, y3 = map(float, input().split())  # (1): x1=0.0, y1=0.0, x2=2.0, y2=0.0, x3=2.0, y3=2.0
    c = (x1-x2)**2 + (y1-y2)**2  # (2): c=4.0
    a = (x2-x3)**2 + (y2-y3)**2  # (3): a=4.0
    b = (x3-x1)**2 + (y3-y1)**2  # (4): b=8.0
    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)  # (5): s=64.0
    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s  # (6): px=1.0
    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s  # (7): py=1.0
    ar = a**0.5  # (8): ar=2.0
    br = b**0.5  # (9): br=2.828427
    cr = c**0.5  # (10): cr=2.0
    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5  # (11): r=1.414214
    print("!{:>.3f}".format(px),"{:>.3f}".format(py),"{:>.3f}".format(r))  # (12): NO CHANGE
    
