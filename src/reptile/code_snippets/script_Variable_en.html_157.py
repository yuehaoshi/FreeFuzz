* example 1:

input: [B, ..., M, K], mat2: [B, ..., K, N]
out: [B, ..., M, N]

* example 2:

input: [B, M, K], mat2: [B, K, N]
out: [B, M, N]

* example 3:

input: [B, M, K], mat2: [K, N]
out: [B, M, N]

* example 4:

input: [M, K], mat2: [K, N]
out: [M, N]

* example 5:

input: [B, M, K], mat2: [K]
out: [B, M]

* example 6:

input: [K], mat2: [K]
out: [1]