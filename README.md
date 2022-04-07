# geniusChong
天才冲冲的毕设


# Generator
using Activation function relu&sigmod


# Discriminator
using Activation function relu&sigmod



# Training process

```bash
for number of training iterations do
	for i steps do
		The noise is sampled from the Gaussian distribution and input into the generator
		train the discriminator using the output of the generator
	end for
	train the generator 
end for
```


