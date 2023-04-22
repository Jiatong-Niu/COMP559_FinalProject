# COMP559_FinalProject
Final Project of the COMP559

This project was originally aimed for C++. However, too many external library are reqiured for FEM and Shape Matching implementation. Thus the final version of the project is developed in python.

# How to run
I strongly recommend run the project with pyCharm as it would help to install packages required.

Packages included in this project：

### FEM: 

     matplotlib.pyplot

     time
     
     tkinter
     
### Shape Maching:
 
     taichi 
     
     math
     
After installing all the packages, run FEM.py for visualizing FEM demo, run shapematching.py for shape matching demo.

# Test cases
### FEM: 
You can create your own truss demo by changing the initial parameter. Notice that pyCharm ->Setting ->Python Scientific, click the show plots in tool window. This can provide a animated effect in pycharm.

Changable Parameter: 

    Node_List
    
    Boundary (-1 represent fix, 1 represent free) (Note: if too less constaint, the truss demo can not analysis and can not provide a solution)
    
    Fext (externel force)
    
    E
    
    A
    
(Above are the parameters that are tested to be save to change. I did not fully test the FEM implementation.) Additionally, current matplot grid is x(0,2) y(-1,1). If want a bigger grid manually change the value before time.sleep()

### Shape Matching: 
You can create your own object with particles similar to the way the current test cases are created. You can also test 4 of the existing test cases by change the parameter test_case.
(Note: if you want to create a new test case find a proper time step for avoiding explotion and add the call of init() in main function)
