def LS_analytic_test():
    LS_analytic_error_ave = [];
    for i in range(10):
        LS_analytic_error_ave . append(LS_analytic('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1)[1])
    print LS_analytic_error_ave
    print reduce(lambda x, y: x+y , LS_analytic_error_ave) / len (LS_analytic_error_ave)

def LS_gradient_test():
    LS_gradient_error_ave = [];
    for i in range(10):
        LS_gradient_error_ave.append(LS_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1,1000, 0.3)[1])
    print LS_gradient_error_ave
    return reduce(lambda x, y: x+y , LS_gradient_error_ave) / len (LS_gradient_error_ave)


def LG_gradient_test():
    LG_gradient_error_ave = [];
    for i in range(10):
        LG_gradient_error_ave.append(LG_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.001)[2])
    print LG_gradient_error_ave
    return reduce(lambda x, y: x+y , LG_gradient_error_ave) / len (LG_gradient_error_ave)

def LG_gradient_back_test():
    LG_gradient_error_ave = [];
    for i in range(10):
        LG_gradient_error_ave.append(LG_gradient_back('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.001, 0.5,0.9)[2])
    print LG_gradient_error_ave
    return reduce(lambda x, y: x+y , LG_gradient_error_ave) / len (LG_gradient_error_ave)