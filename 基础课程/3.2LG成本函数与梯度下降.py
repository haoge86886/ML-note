"""""
成本函数
sigmoid不能用平方误差成本,有太多局部最小值,不是凸函数
使用对数似然函数:
          L(f(xi),yi) = { -ln(f(xi))      yi = 1
                        { -ln(1-f(xi))    yi = 0
或:        L(f(xi),yi) = -yi * ln(f(xi)) - (1-yi) * ln(1-f(xi))
则可知:    J(w,b) = 1/m ∑ [L(f(xi),yi)] = -1/m ∑ [yi*ln(f(xi))+(1-y)*ln(1-f(xi))]


梯度下降
wj = wj - α*∂J(w,b)/∂wj
b = b - α*∂J(w,b)/∂b

"""""