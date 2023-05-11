# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
from keyboardAgents import KeyboardAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint
import game

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    debug_choices = {'Stop': 0, 'North': 0, 'West': 0, 'East': 0, 'South': 0}

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        actions.remove("Stop")  # stopping just seems useless
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start), f'enemySide: {self.on_enemy_side(gameState)}')

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        print(f"Best actions: {bestActions}\nValues:{values}\n\n")
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return bestActions[0] # changed from choosing random choice to choosing best choice

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def on_enemy_side(self, gameState):
        if self.red:
            if gameState.data.food.width / 2 < gameState.getAgentState(self.index).getPosition()[0]:
                return True
        else:
            if gameState.data.food.width / 2 > gameState.getAgentState(self.index).getPosition()[0]:
                return True
        return False


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    RUNNING = False
    CAUTIOUS = False
    SAFE = True
    carry_threshold = 6
    caution_distance = 6
    safe_distance = 10
    panic_distance = 4

    def collect_cautiously_feat(self, features, caution_increase, enemy_distance, food_distance):
        features['distanceToFood'] = food_distance
        features['distanceToEnemy'] = enemy_distance * caution_increase
        return features

    def collect_freely_feat(self, features, food_list, minDistance):
        features['distanceToFood'] = minDistance
        features['successorScore'] = - \
            len(food_list)  # self.getScore(successor)
        features['distanceToEnemy'] = 0
        return features

    def return_to_base_feat(self, feat, gameState, myPos, successor, min_distance, action):
        acts = successor.getLegalActions(self.index)
        # this will help us favor paths which have more choices available to them
        futureChoices = len(acts)
        self.debug_choices[action] = futureChoices
        if self.red:
            feat['successorScore'] = - \
                abs((gameState.data.food.width / 2 - 10) - myPos[0])
            feat['distanceToEnemy'] = min_distance
            feat['choicesAvailable'] = futureChoices
        else:
            feat['successorScore'] = - \
                abs((gameState.data.food.width / 2 + 10) - myPos[0])
            feat['distanceToEnemy'] = min_distance
            feat['choicesAvailable'] = futureChoices
        return feat

    def getFeatures(self, gameState, action):
        # initial setup for feature functions
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        food_list = self.getFood(successor).asList()
        enemy_list = self.getOpponents(successor)
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = None
        min_distance = min([self.getMazeDistance(myPos, successor.getAgentState(enemy).getPosition()) for enemy in enemy_list])

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in food_list])
            numCarrying = gameState.data.agentStates[self.index].numCarrying

        # Compute distance to the nearest enemy
        if numCarrying > self.carry_threshold:  # we check to see if we have enough food to return
            self.RUNNING = True
            self.CAUTIOUS = False
            self.SAFE = False
            return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
        if self.on_enemy_side(gameState):
            features['distanceToEnemy'] = min_distance
            if min_distance > self.safe_distance:
                self.RUNNING = False
                self.CAUTIOUS = False
                self.SAFE = True
            elif min_distance > self.caution_distance:
                self.RUNNING = False
                self.CAUTIOUS = True
                self.SAFE = False
            elif min_distance < self.panic_distance:  # Book it
                self.RUNNING = True
                self.CAUTIOUS = False
                self.SAFE = False
            if self.RUNNING:
                return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
            elif self.SAFE:
                return self.collect_freely_feat(features, food_list, minDistance)
            elif self.CAUTIOUS:
                return self.collect_cautiously_feat(features, 2, min_distance, minDistance)
        else:
            self.RUNNING = False
            self.CAUTIOUS = False
            self.SAFE = True
            features['distanceToEnemy'] = 0
            return self.collect_freely_feat(features, food_list, minDistance)

    def getWeights(self, gameState, action):
        if self.RUNNING:
            return {'successorScore': 50, 'distanceToFood': -1, 'choicesAvailable': 60, 'distanceToEnemy': 60, 'crossSides': 1000}
        elif self.CAUTIOUS:
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 20}
        elif self.SAFE:
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    distance_to_middle = -1
    CHASING = False
    SCARED = False
    moving_to_edge = False
    at_edge_waiting = False

    def scaredFeat(self, gameState, myPos, action, invaderDistance, scaredTimer):
        features = util.Counter()

        # if invader distance greater than scared threshold and timer less than threshold
            # stop value should be max
            # features["stop"] = action

        features["invaderDistance"] = invaderDistance # want to max this value if enemy close and timer large


        return features

    def goToMiddleFeat(self, gameState, myPos):
        features = util.Counter()
        distance_to_middle = self.getDistanceToMiddle(gameState, myPos)

        print(f"dist to mid: {distance_to_middle}")

        features["distanceToMiddle"] = distance_to_middle
        return features

    def getDefaultFeatures(self, num_invaders, invaderDistance, reverse):
        features = util.Counter()
        features["numInvaders"] = num_invaders
        features["invaderDistance"] = invaderDistance
        features["reverse"] = reverse
        return features

    def getFeatures(self, gameState, action):
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        ''' feature values '''
        invaders = self.getInvaders(successor)
        invaderDistance = self.getClosestEnemy(myPos, invaders)
        self.SCARED = myState.scaredTimer > 0
        self.CHASING = True if (len(invaders) and not self.SCARED) else False
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        reverse = 1 if action == rev else -1


        if (self.CHASING):
            return self.getDefaultFeatures(len(invaders), invaderDistance, reverse)
        elif (self.SCARED):
            print("Scared, should be running")
            return self.scaredFeat(gameState, myPos, action, invaderDistance, myState.scaredTimer)
        else:
            return self.goToMiddleFeat(gameState, myPos)
    
    def getWeights(self, gameState, action):
        if (self.CHASING):
            return {'numInvaders': -1000, 'invaderDistance': -10, 'reverse': -2}
        elif (self.SCARED):
            return {'invaderDistance': -10, "stop": -2}
        else: 
            return {"distanceToMiddle": -10}

    def getInvaders(self, successor): # return list of enemy agents on our side
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        return invaders

    def getClosestEnemy(self, myPos, invaders):
        dists = []
        if len(invaders) > 0: # if enemy on our side
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        return min(dists) if len(dists) else -1
    
    def getDistanceToMiddle(self, gameState, myPos):
        middle_x = 0
        if self.red:
            middle_x = gameState.data.food.width / 2 - 10
        else:
            middle_x = gameState.data.food.width / 2 + 10

        return self.getMazeDistance(myPos, (middle_x, (gameState.data.food.height / 2 + 1))) # make sure y is legal
    
        # for y in range(int(myPos[1]), int(gameState.data.food.height)):
        #     try:
        #         return self.getMazeDistance(myPos, (middle_x, y))
        #     except:
        #         print("invalid distance")

