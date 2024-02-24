package com.github.coderodde.hmm;

public enum HiddenMarkovModelStateType {
    
    START,
    HIDDEN,
    END;
    
    @Override
    public String toString() {
        switch (this) {
            case START:
                return "S";
                
            case HIDDEN:
                return "H";
                
            case END:
                return "E";
                
            default:
                throw new EnumConstantNotPresentException(
                        HiddenMarkovModelStateType.class, 
                        "Unknown enum constant: " + this);
        }
    }
}
