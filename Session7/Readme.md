## Receptive field calculation for Inception model

| Layer | Feature map | Receptive Field |Jump |
| --- | --- | --- | --- |
| Conv(7x7) S=2 | 112x112x64 | 7x7 | 1 |
| MaxPool(3x3) S=2 | 56x56 | 11x11 | 2 |
| Conv(3x3) S=1 | 56x56 | 19x19 | 4 |
| MaxPool(3x3) S=2 | 28x28 | 27x27 | 4 | 
| 3(a) Conv(5x5) S=1 | 28x28 | 59x59 | 8 |
| 3(b) Conv(5x5) S=1 | 28x28 | 91x91 | 8 |
| MaxPool(3x3) S=2 | 14x14 | 107x107 | 8 |
| 4(a) Conv(5x5) S=1 | 14x14 | 171x171 | 16 |
| 4(b) Conv(5x5) S=1 | 14x14 | 235x235 | 16 |
| 4(c) Conv(5x5) S=1 | 14x14 | 299x299 | 16 |
| 4(d) Conv(5x5) S=1 | 14x14 | 364x364 | 16 |
| 4(e) Conv(5x5) S=1 | 14x14 | 427x427 | 16 |
| MaxPool(3x3) S=2 | 7x7 | 459x459 | 16 |
| 5(a) Conv(5x5) S=1 | 7x7 | 587x587 | 32 |
| 5(b) Conv(5x5) S=1 | 7x7 | 715x715 | 32 |
| Conv(7x7) S=2 | 1x1 | 907x907 | 32 |
