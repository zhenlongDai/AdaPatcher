#include <iostream>

int foo(int a) {
    int b = a * 2;
    return b;
}

int main() {
    int x = 5;
    int b = foo(x);
    int c = b + x;
    if (b > x){
        c = foo(b);
        c = c + x;
    }
    return 0;
}
