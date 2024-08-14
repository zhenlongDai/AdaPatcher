import sys

for i in range(int(input())):  
    x1, y1, x2, y2, x3, y3 = map(float, input().split()) 
    c = (x1-x2)**2 + (y1-y2)**2 
    a = (x2-x3)**2 + (y2-y3)**2 
    b = (x3-x1)**2 + (y3-y1)**2 
    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c) 
    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s 
    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s 
    ar = a**0.5 
    br = b**0.5 
    cr = c**0.5 
    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5 
    print("!{:>.3f}".format(px),"{:>.3f}".format(py),"{:>.3f}".format(r)) 
    
