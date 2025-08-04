# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util


from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

# -----------------------------Q1---------------------------------#
    def evaluationFunction(self, currentGameState, action):
        """
        A reflex evaluation function for Q1.
        Considers: successor score, distance to closest food, ghost proximity,
        capsule count, and scared times.
        """
        successorGameState = currentGameState.generateSuccessor(0, action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [gs.scaredTimer for gs in newGhostStates]

        score = successorGameState.getScore()

        # 1. Food: closer is better (use reciprocal)
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / (minFoodDist + 1)  # avoid div by zero

        # 2. Ghosts: avoid active ghosts, chase scared ones
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:
                if dist <= 1:
                    return -float('inf')  # immediate death, worst move
                # penalize proximity to active ghosts
                score -= 2.0 / (dist)
            else:
                # reward getting closer to scared ghosts (can eat them)
                score += 1.5 / (dist + 1)

        # 3. Capsules: fewer remaining is better (encourage eating)
        capsules = successorGameState.getCapsules()
        score -= 0.5 * len(capsules)

        # 4. Bonus for eating food (implicit via successor score)

        return score
#-----------------------------Q1---------------------------------#
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
#--------------------------Q2-------------------------------#
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth.
        """

        def minimax(state, agentIndex, depth):
            # Terminal or depth cutoff
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Pacman (maximizer)
            if agentIndex == 0:
                bestValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, 1, depth)  # next is first ghost
                    if val > bestValue:
                        bestValue = val
                return bestValue

            # Ghosts (minimizers)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    # Last ghost: next is Pacman, decrease depth
                    nextAgent = 0
                    nextDepth = depth - 1

                bestValue = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, nextAgent, nextDepth)
                    if val < bestValue:
                        bestValue = val
                return bestValue

        # Root: choose the best action for Pacman
        bestScore = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 1, self.depth)  # first ghost next
            if value > bestScore or bestAction is None:
                bestScore = value
                bestAction = action

        return bestAction
#--------------------------Q2-------------------------------#
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

# --------------------------Q3-------------------------------#
    def getAction(self, gameState):
        def alphabeta(state, agentIndex, depth, alpha, beta):
            # Terminal or depth cutoff
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Pacman: maximizer
            if agentIndex == 0:
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphabeta(successor, 1, depth, alpha, beta)
                    if val > value:
                        value = val
                    if value > beta:
                        return value  # beta cutoff
                    alpha = max(alpha, value)
                return value

            # Ghosts: minimizers
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth = depth - 1

                value = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphabeta(successor, nextAgent, nextDepth, alpha, beta)
                    if val < value:
                        value = val
                    if value < alpha:
                        return value  # alpha cutoff
                    beta = min(beta, value)
                return value

        # Root: choose best action for Pacman with initial bounds
        bestScore = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 1, self.depth, alpha, beta)
            if value > bestScore or bestAction is None:
                bestScore = value
                bestAction = action
            if bestScore > beta:
                break  # optional, though root beta is +inf so won't trigger
            alpha = max(alpha, bestScore)

        return bestAction
#--------------------------Q3-------------------------------#
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

# --------------------------Q4-------------------------------#
    def getAction(self, gameState):
        def expectimax(state, agentIndex, depth):
            # Terminal or depth cutoff
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Pacman: max node
            if agentIndex == 0:
                bestValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = expectimax(successor, 1, depth)
                    if val > bestValue:
                        bestValue = val
                return bestValue

            # Ghosts: chance nodes -> expected value
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth = depth - 1

                legalActions = state.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(state)

                total = 0.0
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = expectimax(successor, nextAgent, nextDepth)
                    total += val
                return total / len(legalActions)

        # Root: choose best action for Pacman
        bestScore = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, self.depth)
            if value > bestScore or bestAction is None:
                bestScore = value
                bestAction = action

        return bestAction
# --------------------------Q4-------------------------------#
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
