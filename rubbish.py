###############################################################################

    def place_phase(self):
        danger_result = self.in_danger(self.player_colour)
        if danger_result != None:
            return danger_result

        kill_result = self.in_danger(self.opp_colour)
        if kill_result != None:
            return kill_result

        for i in range(len(self.place_moves)):
            if (self.state[self.place_moves[i][0],
                           self.place_moves[i][1]] == UNOCC and
                not self.node.is_eliminated(self.state,
                                            self.place_moves[i][0],
                                            self.place_moves[i][1],
                                            self.player_colour)):

                return self.place_moves[i]
