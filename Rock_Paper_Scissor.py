import random
guesses=["rock","paper","scissor","quit","score"]
comp=0
you=0
print("---------------------ROCK PAPER SCISSOR GAME---------------------")
print("Rules \n1.If you want to check Score type \"Score\" in your Guess\n2.If you wanna quit then type \"Quit\" in your guess\n3.Type according to choice in Rock,Paper,Scissor respectively ")
while(True):
 guess=input("Enter Your Choice:-").lower()
 chk=random.randint(0,2)
 if(guess==guesses[chk]):
  you+=1
 elif(guess==guesses[3]):
  if(you>comp):
   print("You Won!!")
  elif(you==comp):
   print("Scores Level!!")
  else:
   print("You Lose!!")
  break
 elif(guess not in guesses):
  print("Invalid Input")
  continue
 elif(guess==guesses[4]):
  print(f"Your Score :-{you}")
  print(f"Computer's Score:-{comp}")