# importing dependencies
from distutils.command.config import config
import pygame 
import neat
import time
import os
import random
pygame.font.init()
    
# definging window size
WIN_WIDTH = 500
WIN_HEIGHT = 800

# importing the images 

# transforming images to twice the size 
birdImage = [pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird1.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird2.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird3.png')))]
pipeImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','pipe.png')))
skyImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bg.png')))
groundImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','base.png')))

STAT_FONT = pygame.font.SysFont("comicsans", 50)


#Bird Class

class Bird:
    imgs = birdImage
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        #when we last jumped
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.imgs[0]

    def jump(self):
        self.vel = -10.5 #because in pygame the coordainates of left corner is (0,0)
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        #Calculating the displacement   
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        #setting terminal velocity
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2  

        self.y =self.y + d

        if d < 0 or self.y <self.height + 50 :
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION

        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
        
    def draw(self, win):
            self.img_count += 1 

            if self.img_count < self.ANIMATION_TIME :
                self.img = self.imgs[0]
            elif self.img_count < self.ANIMATION_TIME*2:
                self.img =self.imgs[1]
            elif self.img_count < self.ANIMATION_TIME*3:
                self.img =self.imgs[2]
            elif self.img_count < self.ANIMATION_TIME*4:
                self.img =self.imgs[1]
            elif self.img_count == self.ANIMATION_TIME*4 + 1:
                self.img =self.imgs[0]
                self.img_count = 0

            
            if self.tilt <= -80:
                self.img = self.imgs[1]
                self.img_count = self.ANIMATION_TIME*2


            rotated_image = pygame.transform.rotate(self.img, self.tilt)
            #rotate image around center
            new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft =  (self.x, self.y)).center)
            win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
            #detect collision
            return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipeImage, False, True)
        self.PIPE_BOTTOM = pipeImage

        #for collision 
        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        #masks for bird
        bird_mask = bird.get_mask()

        #masks for pipes
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        #calculate offset (distance between masks)
        #Top pipe offset
        top_offset = (self.x - bird.x, self.top - round(bird.y))

        #Bottom pipe offset
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        #Check collision NON for False   
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
            #We are colliding
        return False

class Base:
    VEL = 5
    WIDTH = groundImage.get_width()
    IMG = groundImage

    def __init__(self, y):
        self.y = y
        #image 1
        self.x1 = 0
        #image 2
        self.x2 = self.WIDTH

    def move(self): 

        #Moving with velocity VEL to the left
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        
        #putting image 1 behind image 2
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        #putting image 2 behind image 1
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):

        #Display ground
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

    


def draw_window(win, birds, pipes, base, score):

    win.blit(skyImage, (0,0))
    
    for pipe in pipes:
        pipe.draw(win)
    
    text = STAT_FONT.render("Score: "+ str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH-10-text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()

def main(genomes, config):

    nets = []
    ge = []
    birds = []

    for _, g in genomes:#genome is a tupple with key and object and we just need the object
        #setting up neural network for each bird 
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run =False
                pygame.quit()
                quit()
                
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

        #Activate neural network with inputs to input layer
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False          
        rem =[]
        for pipe in pipes:
            #Check for collision
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    #reduce fitness score on collision
                    ge[x].fitness -= 1
                    #remove bird from the lists 
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x) 

                #Check if bird has passed the pipe   
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            #Check if pipe is off the screen    
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        #Add new pipe 
        if add_pipe:
            score += 1

            #increase the fitness score of passed pipe
            for g in ge:
                g.fitness += 5

            pipes.append(Pipe(700))
        
        #Remove previous pipes
        for r in rem: 
            pipes.remove(r)

        for x, bird in enumerate(birds):
            #Check if bird hits the ground
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)


        base.move() 
        draw_window(win, birds, pipes, base, score)
  



def run(config_path):
    #Define all subheadings used in the config file 
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    #Create Population 
    p = neat.Population(config)

    #Print stats of the training
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    #Set the fitness function, No. of generations
    winner = p.run(main,50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_neuralnet.txt")
    run(config_path)