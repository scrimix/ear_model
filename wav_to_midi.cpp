#include "note_model.h"

int main()
{
    note_model_params_t params;
    params.models_path = "../../models/carfac_notes";
    
    note_model_t model;
    model.setup(params);
    model.load();
}