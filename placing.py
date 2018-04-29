    def extpos(self, x):
        return x[0], x[1]

    def place_phase(self):

        if self.state[extpos(placing_moves[0])] == UNOCC:
            return placing_moves[0]
        if self.state[extpos(placing_moves[1])] == UNOCC:
            return placing_moves[1]

        while self.totalTurns < 21:
            if self.state[extpos(placing_moves[2])] == UNOCC and self.isEliminated(self.state, extpos(placing_moves[2]), self.colour) == False:
                return placing_moves[2]
            if self.state[extpos(placing_moves[3])] == UNOCC and self.isEliminated(self.state, extpos(placing_moves[3]), self.colour) == False:
                return placing_moves[3]
            if self.state[extpos(placing_moves[2])] == self.colour and self.state[extpos(placing_moves[4])] == UNOCC and self.isEliminated(self.state, extpos(placing_moves[4]), self.colour) == False:
                return placing_moves[4]
            if self.state[extpos(placing_moves[3])] == self.colour and self.state[extpos(placing_moves[5])] == UNOCC and self.isEliminated(self.state, extpos(placing_moves[5]), self.colour) == False:
                return placing_moves[5]
