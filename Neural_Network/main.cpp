#include <SFML/Graphics.hpp>
#include "FFNN/FFNN.hpp"
#include "Classifier/TrainerClassifier.hpp"
#include "Classifier/Scope.hpp"
#include "Dataset/Dataset.hpp"

using namespace sf;

hyperparameters hyper = {
    input_dim : 28*28,
    output_dim : 10,
    hidden_layer_sizes : { 256, 128 },
    learning_rate : 0.001,
    dropout_rate : 0.2,
    max_epochs : 50,
    n_train_samples : 10000,
    mini_batch_size : 32,
    n_val_samples : 1000,

    early_stopping : true,
    patience : 10
};

int main() {
    FFNN model(hyper);

    bool learning = false;
    print("Train ? (y/n)"); char a; std::cin >> a;
    if (a == 'y') learning = true;

    bool store = true;

    if(learning) {

        Scope scope(model, hyper);
    
        TrainerClassifier trainer(model, hyper);

        Dataset train = DataLoader(hyper, "train");
        Dataset validation = DataLoader(hyper, "validation");

        trainer.set_scope(scope);
        trainer.set_data(train, validation);
        print("Data has been successfully imported");

        trainer.run(store);
        model.saveWeights("executable/model_weights.txt");
        print("Weights saved !");

    } else {

        model.loadWeights("executable/model_weights.txt");
        print("Weights loaded !");

    }


    // Window init
    sf::RenderWindow window(sf::VideoMode({ 800, 800 }), "Deep Learning with Adam Optimizer");
    window.setFramerateLimit(100);
    sf::View view({ 14, 14 }, { 30, 30 });
    window.setView(view);

    // Canvas init
    sf::RenderTexture canvas({ 28, 28 });
    canvas.clear(sf::Color::White);
    sf::Sprite sprite(canvas.getTexture());


    // Cursor init
    RectangleShape cursor({ 2, 2 });
    cursor.setFillColor(Color(255, 255, 255, 0));
    cursor.setOutlineThickness(0.5);
    cursor.setOrigin({ cursor.getSize().x / 2, cursor.getSize().y / 2 });
    cursor.setOutlineColor(Color(0, 0, 0));
    // Brush border with 20% opacity
    const float brush_size = 0.75;
    sf::CircleShape brush(brush_size * 2, 5);
    brush.setOrigin({ brush_size * 2, brush_size * 2 });
    brush.setFillColor(Color(120, 0, 255, 50));
    // Brush center with 100% opacity
    sf::CircleShape brushCenter(brush_size, 5);
    brushCenter.setOrigin({ brush_size, brush_size });
    brushCenter.setFillColor(Color(120, 0, 255, 255));

    // Main loop
    bool firstPress = true;
    while (window.isOpen())
    {
        Vector2f mousePos = window.mapPixelToCoords(Mouse::getPosition(window));
        cursor.setPosition(mousePos);

        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                // "Escape" closes the window
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();

                // "R" resets the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
                    canvas.clear(sf::Color::White);
                    canvas.display();
                }

                // "A" gives number prediction from the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::A) {
                    if (firstPress) {
                        Matrix pixels(1, 28 * 28);
                        for (size_t i = 0; i < 28; i++)
                            for (size_t j = 0; j < 28; j++)
                                pixels(0, j * 28 + i) = 1.f - static_cast<int>(canvas.getTexture().copyToImage().getPixel({ i, j }).g) / 255.f;

                        model.forward(pixels);
                        print("The number you've drawn is ", model.getOutput().getMaxIndex(), " !!!");
                    }
                    firstPress = false;
                }
            }
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyReleased>())
                if (keyPressed->scancode == sf::Keyboard::Scancode::A)
                    firstPress = true;
        }

        // If I left click, it draws
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            if (mousePos.x + 1 < 28 && mousePos.x > 1) {
                if (mousePos.y + 1 < 28 && mousePos.y > 1) {
                    brushCenter.setPosition(mousePos);
                    brush.setPosition(mousePos);
                    canvas.draw(brushCenter);
                    canvas.draw(brush);
                    canvas.display();
                }
            }
        }


        // Updating each frame
        window.clear(sf::Color(64, 64, 64));
        window.draw(sprite);
        window.draw(cursor);
        window.display();
    }

    return 0;
}
