import turtle

# configurar el tamaño de la ventana de turtle
turtle.setup(800, 800)

# dibujar un cuadrado
t = turtle.Turtle()
t.speed(0)
t.penup()
t.goto(-200, -200)
t.pendown()
for i in range(4):
    t.forward(400)
    t.left(90)

# asegurarse de que la ventana permanezca abierta
turtle.done()
