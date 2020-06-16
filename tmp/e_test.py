
from math import sin, cos


g = 2
l = 3
m_p = 4
m_c = 5

x_acc = 6
y_acc = 7
t_ang = 8
t_vel = 9
t_acc = 10

f_x = 11
f_y = 12

t_acc = f_x * cos(t_ang) + f_y * sin(t_ang) + 2 * m_p * l * t_vel * t_vel * sin(t_ang) * cos(t_ang)
t_acc /= m_p * l * (sin(t_ang) * sin(t_ang) + cos(t_ang) * cos(t_ang)) - (m_c + m_p) * l

#a = (f_x * cos(t_ang) + f_y * sin(t_ang)) / ((m_c + m_p) * l)
#b = m_p / (m_c + m_p) * t_vel * t_vel * (sin(t_ang) + cos(t_ang))
#c = 1 - m_p / (m_c + m_p) * (sin(t_ang) + cos(t_ang))
#t_acc = -(a + b) / c


print("t_acc", t_acc)


x_acc =  (f_x - m_p * l * (t_acc * cos(t_ang) - t_vel * t_vel * sin(t_ang))) / (m_p + m_c)
y_acc =  (f_y - g * (m_p + m_c) - m_p * l * (t_acc * sin(t_ang) - t_vel * t_vel * cos(t_ang))) / (m_p + m_c)

t_acc = (-g * sin(t_ang) - x_acc * cos(t_ang) - y_acc * sin(t_ang)) / l

print("x_acc", x_acc)
print("y_acc", y_acc)
print("t_acc", t_acc)







#print("x_acc", x_acc)
#print("y_acc", y_acc)
