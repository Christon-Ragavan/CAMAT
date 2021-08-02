import numpy as np
import matplotlib.pyplot as plt
def my_decorator_arg(name_2):
    def my_decorator(func, *args, **kwargs):
        def wrapper(*args, **kwargs):
            print("Something is happening before the function is called.")
            r = func(*args, **kwargs)
            print("return" ,r)
            print("Something is happening after the function is called.")
        return wrapper
    return my_decorator

@my_decorator_arg(name_2="NAME2CHOED")
def say_whee(name):
    print("Whee!", name)
    return "I RETUTN "
say_whee(name='Cjrostp')

#
# fig = plt.figure( figsize=(8, 4 ) )
# ax = fig.add_axes([.05, .1, .9, .85 ] )
# ax.set_yticks( np.linspace(0, 200, 11 ) )
#
# xticks = [1,  2, 3, 4, 6, 8, 10 ]
# xticks_minor = [ 1, 5, 7, 9, 11 ]
# xlbls = [ '1', 'b', '5yr', 'fb',
#           'maximum day', '90th percentile day', 'average day' ]
#
# ax.set_xticks( xticks )
# ax.set_xticks( xticks_minor, minor=True )
# ax.set_xticklabels( xlbls )
# ax.set_xlim( 1, 11 )
#
# ax.grid( 'off', axis='x' )
# ax.grid( 'off', axis='x', which='minor' )
#
# # vertical alignment of xtick labels
# va = [ 0, -.05, 0, -.05, -.05, -.05 ]
# for t, y in zip( ax.get_xticklabels( ), va ):
#     t.set_y( y )
#
# ax.tick_params( axis='x', which='minor', direction='out', length=30 )
# ax.tick_params( axis='x', which='major', bottom='off', top='off' )
#
# plt.show()