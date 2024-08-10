# sentencepiece can be used for both training and inference
# this scripts aims at training a tokenizer

import sentencepiece as spm
import os
# Creating a txt file
with open("toy.txt", "w", encoding="utf-8") as f:
    f.write("""IT was seven o'clock of a very warm evening in the Seeonee hills when
Father Wolf woke up from his day's rest, scratched himself, yawned, and
spread out his paws one after the other to get rid of the sleepy feeling
in the tips. Mother Wolf lay with her big gray nose dropped across her
four tumbling, squealing cubs, and the moon shone into the mouth of the
cave where they all lived. "Augrh!" said Father Wolf, "it is time to
hunt again"; and he was going to spring downhill when a little shadow
with a bushy tail crossed the threshold and whined: "Good luck go with
you, O Chief of the Wolves; and good luck and strong white teeth go with
the noble children, that they may never forget the hungry in this
world."

   [Illustration: "'GOOD LUCK GO WITH YOU, O CHIEF OF THE WOLVES.'"]

It was the jackal--Tabaqui, the Dish-licker--and the wolves of India
despise Tabaqui because he runs about making mischief, and telling
tales, and eating rags and pieces of leather from the village
rubbish-heaps. They are afraid of him too, because Tabaqui, more than
any one else in the jungle, is apt to go mad, and then he forgets that
he was ever afraid of any one, and runs through the forest biting
everything in his way. Even the tiger hides when little Tabaqui goes
mad, for madness is the most disgraceful thing that can overtake a wild
creature. We call it hydrophobia, but they call it _dewanee_--the
madness--and run.

"Enter, then, and look," said Father Wolf, stiffly; "but there is no
food here."

"For a wolf, no," said Tabaqui; "but for so mean a person as myself a
dry bone is a good feast. Who are we, the Gidur-log [the Jackal People],
to pick and choose?" He scuttled to the back of the cave, where he found
the bone of a buck with some meat on it, and sat cracking the end
merrily.

"All thanks for this good meal," he said, licking his lips. "How
beautiful are the noble children! How large are their eyes! And so
young too! Indeed, indeed, I might have remembered that the children of
kings are men from the beginning."

Now, Tabaqui knew as well as any one else that there is nothing so
unlucky as to compliment children to their faces; and it pleased him to
see Mother and Father Wolf look uncomfortable.

Tabaqui sat still, rejoicing in the mischief that he had made, and then
he said spitefully:

"Shere Khan, the Big One, has shifted his hunting-grounds. He will hunt
among these hills during the next moon, so he has told me."

Shere Khan was the tiger who lived near the Waingunga River, twenty
miles away.

"He has no right!" Father Wolf began angrily. "By the Law of the Jungle
he has no right to change his quarters without fair warning. He will
frighten every head of game within ten miles; and I--I have to kill for
two, these days."

"His mother did not call him Lungri [the Lame One] for nothing," said
Mother Wolf, quietly. "He has been lame in one foot from his birth. That
is why he has only killed cattle. Now the villagers of the Waingunga are
angry with him, and he has come here to make _our_ villagers angry.
They will scour the jungle for him when he is far away, and we and our
children must run when the grass is set alight. Indeed, we are very
grateful to Shere Khan!"

"Shall I tell him of your gratitude?" said Tabaqui.

"Out!" snapped Father Wolf. "Out, and hunt with thy master. Thou hast
done harm enough for one night."

"I go," said Tabaqui, quietly. "Ye can hear Shere Khan below in the
thickets. I might have saved myself the message."

Father Wolf listened, and in the dark valley that ran down to a little
river, he heard the dry, angry, snarly, singsong whine of a tiger who
has caught nothing and does not care if all the jungle knows it.

"The fool!" said Father Wolf. "To begin a night's work with that noise!
Does he think that our buck are like his fat Waingunga bullocks?"

"H'sh! It is neither bullock nor buck that he hunts to-night," said
Mother Wolf; "it is Man." The whine had changed to a sort of humming
purr that seemed to roll from every quarter of the compass. It was the
noise that bewilders wood-cutters, and gipsies sleeping in the open,
and makes them run sometimes into the very mouth of the tiger.

"Man!" said Father Wolf, showing all his white teeth. "Faugh! Are there
not enough beetles and frogs in the tanks that he must eat Man--and on
our ground too!"

The Law of the Jungle, which never orders anything without a reason,
forbids every beast to eat Man except when he is killing to show his
children how to kill, and then he must hunt outside the hunting-grounds
of his pack or tribe. The real reason for this is that man-killing
means, sooner or later, the arrival of white men on elephants, with
guns, and hundreds of brown men with gongs and rockets and torches. Then
everybody in the jungle suffers. The reason the beasts give among
themselves is that Man is the weakest and most defenseless of all living
things, and it is unsportsmanlike to touch him. They say too--and it is
true--that man-eaters become mangy, and lose their teeth.

The purr grew louder, and ended in the full-throated "Aaarh!" of the
tiger's charge.

Then there was a howl--an untigerish howl--from Shere Khan. "He has
missed," said Mother Wolf. "What is it?"

Father Wolf ran out a few paces and heard Shere Khan muttering and
mumbling savagely, as he tumbled about in the scrub.

"The fool has had no more sense than to jump at a wood-cutters'
camp-fire, so he has burned his feet," said Father Wolf, with a grunt.
"Tabaqui is with him."

"Something is coming uphill," said Mother Wolf, twitching one ear. "Get
ready."

The bushes rustled a little in the thicket, and Father Wolf dropped with
his haunches under him, ready for his leap. Then, if you had been
watching, you would have seen the most wonderful thing in the world--the
wolf checked in mid-spring. He made his bound before he saw what it was
he was jumping at, and then he tried to stop himself. The result was
that he shot up straight into the air for four or five feet, landing
almost where he left ground.

"Man!" he snapped. "A man's cub. Look!"

Directly in front of him, holding on by a low branch, stood a naked
brown baby who could just walk--as soft and as dimpled a little thing as
ever came to a wolf's cave at night. He looked up into Father Wolf's
face and laughed.

"Is that a man's cub?" said Mother Wolf. "I have never seen one. Bring
it here."

A wolf accustomed to moving his own cubs can, if necessary, mouth an
egg without breaking it, and though Father Wolf's jaws closed right on
the child's back not a tooth even scratched the skin, as he laid it down
among the cubs.

"How little! How naked, and--how bold!" said Mother Wolf, softly. The
baby was pushing his way between the cubs to get close to the warm hide.
"Ahai! He is taking his meal with the others. And so this is a man's
cub. Now, was there ever a wolf that could boast of a man's cub among
her children?"

"I have heard now and again of such a thing, but never in our pack or in
my time," said Father Wolf. "He is altogether without hair, and I could
kill him with a touch of my foot. But see, he looks up and is not
afraid.""")


options = dict(
    input="toy.txt",
    input_format="text",
    model_prefix="tok400", #output filename prefix
    model_type="bpe",
    vocab_size=400,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=10000,
    shuffle_input_sentence=True,
    character_coverage= 0.99995,
    byte_fallback=True, # set to False and observe the results for the text that was not part of the training
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    num_threads=os.cpu_count()
)

spm.SentencePieceTrainer.Train(**options)

sp =spm.SentencePieceProcessor()
sp.load("tok400.model")
vocab =[[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
#print(vocab)

# testing
ids = sp.encode("hello مرحبا")
print(ids)
print([sp.id_to_piece(idx) for idx in ids]) # here you will see bytes for مرحبا because it was not in the training set.