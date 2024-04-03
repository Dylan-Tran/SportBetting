class Game_record(object):
    def __init__(self, player1_id, player2_id, player1_outcome) -> None:
        self.player1 = player1_id
        self.player2 = player2_id

        self.player1_outcome = player1_outcome
        self.player2_outcome = -self.player1_outcome

    def __repr__(self) -> str:
        if self.player1_outcome == self.player2_outcome:
            winner = "tie"
        else:
            winner = "player 1" if self.player1_outcome > 0 else "player 2"

        return f"[player1: {self.player1}, player2: {self.player2}, spread 1: {self.player1_outcome} winner: {winner}]\n"

    def contain_player(self, player_id) -> bool:
        return self.player1 == player_id or self.player2 == player_id

    def get_other_player(self, player_id) -> str:
        if self.player1 == player_id:
            return self.player2
        elif self.player2 == player_id:
            return self.player1
        else:
            raise Exception("Player was not found")

    def get_player_result(self, player_id):
        if self.player1 == player_id:
            return self.player1_outcome
        elif self.player2 == player_id:
            return self.player2_outcome
        else:
            raise Exception("Player was not found")

    @staticmethod
    def get_player1(self) -> str:
        return self.player1
