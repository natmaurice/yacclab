sl_tree_0: if ((c+=2) >= w - 2) { if (c > w - 2) { goto sl_break_0_0; } else { goto sl_break_1_0; } } 
		if (CONDITION_O) {
			if (CONDITION_P) {
				ACTION_2
				goto sl_tree_1;
			}
			else {
				ACTION_2
				goto sl_tree_0;
			}
		}
		else {
			if (CONDITION_P) {
				ACTION_2
				goto sl_tree_1;
			}
			else {
				ACTION_1
				goto sl_tree_0;
			}
		}
sl_tree_1: if ((c+=2) >= w - 2) { if (c > w - 2) { goto sl_break_0_1; } else { goto sl_break_1_1; } } 
		if (CONDITION_O) {
			if (CONDITION_P) {
				ACTION_6
				goto sl_tree_1;
			}
			else {
				ACTION_6
				goto sl_tree_0;
			}
		}
		else {
			if (CONDITION_P) {
				ACTION_2
				goto sl_tree_1;
			}
			else {
				ACTION_1
				goto sl_tree_0;
			}
		}
sl_break_0_0:
		if (CONDITION_O) {
			ACTION_2
		}
		else {
			ACTION_1
		}
		goto sl_;
sl_break_0_1:
		if (CONDITION_O) {
			ACTION_6
		}
		else {
			ACTION_1
		}
		goto sl_;
sl_break_1_0:
		if (CONDITION_O) {
			if (CONDITION_P) {
				ACTION_2
			}
			else {
				ACTION_2
			}
		}
		else {
			if (CONDITION_P) {
				ACTION_2
			}
			else {
				ACTION_1
			}
		}
		goto sl_;
sl_break_1_1:
		if (CONDITION_O) {
			if (CONDITION_P) {
				ACTION_6
			}
			else {
				ACTION_6
			}
		}
		else {
			if (CONDITION_P) {
				ACTION_2
			}
			else {
				ACTION_1
			}
		}
		goto sl_;
sl_:;
