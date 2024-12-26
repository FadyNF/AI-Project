import numpy as np
import math 


totalNumberOfClassifications = int(input('Total Number: '))
p1 = (int(input('First Prob: ')) / totalNumberOfClassifications)
p2 = (int(input('Second Prob: ')) / totalNumberOfClassifications)


entropy = -((p1 * math.log(p1, 2)) + (p2 * math.log(p2, 2)))
print('Entropy: ', entropy)