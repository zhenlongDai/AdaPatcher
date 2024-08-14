def ceil(n, m):
  return n / m + 2 if n % m != 0 else n / m + 1

def gram(l):
  return 200 * l[0] + 300 * l[1] + 500 * l[2]

def price(l):
  def c(n, m, p, r):
    d = (n / m) * m
    return ((n - d) + d * r) * p

  return int(c(l[0], 5, 380, 0.80) + c(l[1], 4, 550, 0.85) + c(l[2], 3, 850, 0.88))

t = []
for amount in range(500, 5500, 100):
  for a in range(ceil(amount, 200)):
    for b in range(ceil(amount, 300)):
      for c in range(ceil(amount, 500)):
        if gram([a, b, c]) == amount:
          t.append([gram([a, b, c]), [a, b, c]])

while True:
  data = input()
  if data == 0:
    break
  s = []
  for x in t:
    if x[0] == data:
      s.append(price(x[1]))
  print sorted(s)[0]