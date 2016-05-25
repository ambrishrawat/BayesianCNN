Exp 6

Investigation of an image that doesn't get misclassified after 10,000 gradient steps

Log:

1. The image X_test[1][:] was investigated (Original Label: Ship)
2. Deteministic gradients are logged for different adversarial labels
	Adversarial Label: airplane
	Adversarial Label: bird

3. It was observed that w.r.t. birds the computed gradients are always 0.000
4. Hypotheses:
	- Numerical errors?
	- Vanishing gradients?
