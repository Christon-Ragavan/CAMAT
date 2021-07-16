
import tkinter as tk

root =tk.Tk()

# setting the windows size
root.geometry("600x400")

# declaring string variable
# for storing name and password
name_var =tk.StringVar()
passw_var =tk.StringVar()


# defining a function that will
# get the name and password and
# print them on the screen
def submit():

    name =name_var.get()
    password =passw_var.get()

    print("The name is : " + name)
    print("The password is : " + password)

    name_var.set("")
    passw_var.set("")


name_label = tk.Label(root, text = 'Username', font=('calibre' ,10, 'bold'))
name_entry = tk.Entry(root ,textvariable = name_var, font=('calibre' ,10 ,'normal'))
passw_label = tk.Label(root, text = 'Password', font = ('calibre' ,10 ,'bold'))
passw_entry =tk.Entry(root, textvariable = passw_var, font = ('calibre' ,10 ,'normal'), show = '*')
sub_btn =tk.Button(root ,text = 'Submit', command = submit)

# placing the label and entry in
# the required position using grid
# method
name_label.grid(row=0 ,column=0)
name_entry.grid(row=0 ,column=1)
passw_label.grid(row=1 ,column=0)
passw_entry.grid(row=1 ,column=1)
sub_btn.grid(row=2 ,column=1)

# performing an infinite loop
# for the window to display
root.mainloop()