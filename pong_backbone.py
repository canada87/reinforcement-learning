import turtle

wn = turtle.Screen()
wn.title('pong')
wn.bgcolor('black')
wn.setup(width=800, height=600)
wn.tracer(0)

class elements:
	def __init__(self):
		obj = turtle.Turtle()
		obj.speed(0)
		obj.color('white')
		obj.penup()
		self.obj = obj

	def paddle(self,x,y):
		self.obj.shape('square')
		self.obj.shapesize(stretch_wid=5, stretch_len=1)
		self.obj.goto(x,y)

	def ball(self,x,y):
		self.obj.shape('square')
		self.obj.goto(x,y)
		self.obj.dx = 0.07
		self.obj.dy = 0.07

	def move_up(self):
		y = self.obj.ycor()
		y += 20
		self.obj.sety(y)

	def move_down(self):
		y = self.obj.ycor()
		y -= 20
		self.obj.sety(y)

	def scoring(self):
		self.obj.hideturtle()
		self.obj.goto(0,260)
		self.obj.write('Player A: 0 Player B: 0', align='center', font=("Courier", 24, "normal"))
		self.scoreA = 0
		self.scoreB = 0

paddle_a = elements()
paddle_b = elements()
palla = elements()
score_bar = elements()

paddle_a.paddle(-350, 0)
paddle_b.paddle(350, 0)
palla.ball(0, 0)
score_bar.scoring()

wn.listen()
wn.onkeypress(paddle_a.move_up, 'w')
wn.onkeypress(paddle_a.move_down, 's')
# wn.onkeypress(paddle_b.move_up, 'i')
# wn.onkeypress(paddle_b.move_down, 'k')

while True:
	wn.update()

	palla.obj.setx(palla.obj.xcor() + palla.obj.dx)
	palla.obj.sety(palla.obj.ycor() + palla.obj.dy)

	paddle_b.obj.sety(palla.obj.ycor())

	#valori che servono alla rete
	# palla.obj.xcor()
	# palla.obj.ycor()
	# paddle_b.obj.ycor()

	if palla.obj.ycor() > 290:
		palla.obj.sety(290)
		palla.obj.dy *=-1

	if palla.obj.ycor() < -290:
		palla.obj.sety(-290)
		palla.obj.dy *=-1

	if palla.obj.xcor() > 390:
		palla.obj.goto(0,0)
		palla.obj.dx *= -1
		score_bar.scoreA += 1
		score_bar.obj.clear()
		score_bar.obj.write('Player A: {} Player B: {}'.format(score_bar.scoreA, score_bar.scoreB), align='center', font=("Courier", 24, "normal"))

	if palla.obj.xcor() < -390:
		palla.obj.goto(0,0)
		palla.obj.dx *= -1
		score_bar.scoreB += 1
		score_bar.obj.clear()
		score_bar.obj.write('Player A: {} Player B: {}'.format(score_bar.scoreA, score_bar.scoreB), align='center', font=("Courier", 24, "normal"))

	if (palla.obj.xcor() > 340 and palla.obj.xcor() < 350)and (palla.obj.ycor() < paddle_b.obj.ycor() + 50 and palla.obj.ycor() > paddle_b.obj.ycor() -50):
		palla.obj.setx(340)
		palla.obj.dx *= -1

	if (palla.obj.xcor() < -340 and palla.obj.xcor() > -350)and (palla.obj.ycor() < paddle_a.obj.ycor() + 50 and palla.obj.ycor() > paddle_a.obj.ycor() -50):
		palla.obj.setx(-340)
		palla.obj.dx *= -1
